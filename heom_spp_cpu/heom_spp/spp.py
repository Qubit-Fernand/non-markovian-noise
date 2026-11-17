"""HEOM -> SPP: one time-MPS core per learning layer, without enumerating histories.

Within a layer the (memory direction alpha, Pauli b) queries are independent
and run on a bounded CPU process pool; layers are sequential because layer j
needs the memory C^{j-1} left by layer j-1. The bath is never reset or closed
between layers; only the last layer reads the root ADO.

build_spp(reset=True) is the layer-wise independent baseline instead: every
layer starts from zero ADOs and reads the root ADO, so the result is a product
of single-layer distributions (time bond 1).


One persistent spawn pool per construction. Incoming bath cores are shared
read-only once per layer, not pickled with every Pauli task. Results are
consumed in deterministic order with bounded in-flight work.
"""
import atexit
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from math import prod
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import os
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from .mps import MPS, Truncation, split
from .pauli import characters, paulis

_ENGINE = _SLOTS = _PAULIS = _LIMITER = None
_SHARED = None
_MEMORY = None
_MEMORY_NAME = None
_THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def available_cpus():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


class SharedCores:
    def __init__(self, cores):
        self.shm = SharedMemory(create=True, size=sum(c.nbytes for c in cores))
        specs, offset = [], 0
        for core in cores:
            a = np.ascontiguousarray(core)
            view = np.ndarray(a.shape, dtype=a.dtype, buffer=self.shm.buf, offset=offset)
            view[:] = a
            specs.append((a.shape, a.dtype.str, offset))
            offset += a.nbytes
        self.description = (self.shm.name, specs)

    def __enter__(self):
        return self.description

    def __exit__(self, *exc):
        self.shm.close()
        self.shm.unlink()


def _close_worker_memory():
    global _MEMORY, _SHARED, _MEMORY_NAME
    _MEMORY = None
    if _SHARED is not None:
        _SHARED.close()
    _SHARED = _MEMORY_NAME = None


def _worker_init(engine, slots, threads):
    global _ENGINE, _SLOTS, _PAULIS, _LIMITER
    for name in _THREAD_KEYS:
        os.environ[name] = str(threads)
    _LIMITER = threadpool_limits(limits=threads)
    _ENGINE, _SLOTS = engine, slots
    _PAULIS = paulis(engine.n)
    if hasattr(engine, "prepare"):
        engine.prepare(slots)
    atexit.register(_close_worker_memory)


def _shared_memory(description):
    global _MEMORY, _MEMORY_NAME, _SHARED
    name, specs = description
    if name != _MEMORY_NAME:
        _close_worker_memory()
        _SHARED = SharedMemory(name=name)
        _MEMORY_NAME = name
        _MEMORY = [np.ndarray(shape, dtype=dtype, buffer=_SHARED.buf, offset=offset)
                   for shape, dtype, offset in specs]
        for core in _MEMORY:
            core.flags.writeable = False
    return _MEMORY


def _compute(engine, slots, basis, cores, job):
    stage, alpha, start, stop = job
    memory = MPS([cores[0][alpha:alpha+1], *cores[1:]])
    before = asdict(engine.stats)
    cpu_before = dict(getattr(engine, "tdvp_stats", {}))
    t = perf_counter()
    responses = [engine.response(slots[stage], basis[b], memory).cores
                 for b in range(start, stop)]
    after = asdict(engine.stats)
    stats = {k: (after[k] if k.endswith("_max") or k == "max_bond_seen"
                 else after[k]-before[k]) for k in after}
    # maxima are lifetime maxima; taking max in the parent is intentional.
    cpu_after = getattr(engine, "tdvp_stats", {})
    cpu = {k: (v if k.startswith("max_") else v-cpu_before[k])
           for k, v in cpu_after.items()}
    return job, responses, stats, cpu, perf_counter()-t


def _worker(job):
    stage, alpha, start, stop, description = job
    return _compute(_ENGINE, _SLOTS, _PAULIS, _shared_memory(description),
                    (stage, alpha, start, stop))


def _bounded_ordered(pool, jobs, maximum):
    jobs = iter(jobs)
    pending = deque()
    for _ in range(maximum):
        try:
            pending.append(pool.submit(_worker, next(jobs)))
        except StopIteration:
            break
    while pending:
        yield pending.popleft().result()
        try:
            pending.append(pool.submit(_worker, next(jobs)))
        except StopIteration:
            pass


def _walsh_tag(core, rank, n):
    m = 4**n
    a = core.reshape(rank, *([4]*n), core.shape[-1])
    h = characters(1)/4
    for axis in range(1, n+1):
        a = np.moveaxis(np.tensordot(h, a, axes=(1, axis)), 0, axis)
    return a.reshape(1, rank*m, core.shape[-1])


class _Assembler:
    def __init__(self, truncation, method):
        self.truncation, self.method = truncation, method
        self.levels, self.result = [], None
        self.discarded = 0.

    def _add(self, left, right):
        result, errors = left.add(right).round(self.truncation)
        self.discarded += sum(errors)
        return result

    def push(self, branch):
        if self.method == "sequential":
            self.result = branch if self.result is None else self._add(self.result, branch)
            return
        level = 0
        while level < len(self.levels) and self.levels[level] is not None:
            branch = self._add(self.levels[level], branch)
            self.levels[level] = None
            level += 1
        if level == len(self.levels):
            self.levels.append(branch)
        else:
            self.levels[level] = branch

    def finish(self):
        if self.method == "sequential":
            return self.result
        result = None
        for a in reversed(self.levels):
            if a is not None:
                result = a if result is None else self._add(result, a)
        return result


def _aggregate(engine, stats, cpu):
    for k, value in stats.items():
        old = getattr(engine.stats, k)
        setattr(engine.stats, k, max(old, value) if k.endswith("_max") or
                k == "max_bond_seen" else old+value)
    if hasattr(engine, "tdvp_stats"):
        for k, value in cpu.items():
            engine.tdvp_stats[k] = max(engine.tdvp_stats[k], value) if k.startswith("max_") \
                else engine.tdvp_stats[k]+value


def build_spp(engine, slots, *, workers=1, blas_threads=1, chunk_size=8,
                  assembly="tree", time_truncation=Truncation(),
                  assembly_truncation=Truncation(), progress=None, reset=False):
    """Same gate-after SPP, with deterministic sequential or tree assembly.

    C(alpha, bath) is shared without closing the bath. Only the last slot
    selects root ADOs. No full physical PTM or full history tensor is built.
    Truncation errors of tree and sequential addition need not be identical.

    reset=True: zero ADOs at the start of every layer and root readout at its
    end (independent layers, what a layer-wise noise model describes).
    """
    slots = tuple(slots)
    if not slots or min(workers, blas_threads, chunk_size) < 1:
        raise ValueError("nonempty slots and positive CPU settings required")
    if workers*blas_threads > available_cpus():
        raise ValueError("workers * blas_threads exceeds the CPU affinity allocation")
    if assembly not in ("tree", "sequential"):
        raise ValueError("assembly must be tree or sequential")
    m, basis = 4**engine.n, paulis(engine.n)
    memory = engine.root().cores
    out, records = [], []
    if hasattr(engine, "prepare"):
        engine.prepare(slots)
    saved_env = {name: os.environ.get(name) for name in _THREAD_KEYS}
    pool = None
    try:
        for name in _THREAD_KEYS:
            os.environ[name] = str(blas_threads)
        if workers > 1:
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                initializer=_worker_init, initargs=(engine, slots, blas_threads))
        with threadpool_limits(limits=blas_threads):
            for stage, slot in enumerate(slots):
                t0 = perf_counter()
                rank = memory[0].shape[0]
                last = reset or stage == len(slots)-1
                final = np.zeros((rank, m), complex) if last else None
                merger = _Assembler(assembly_truncation, assembly)
                worker_sum = wait = assembly_time = 0.
                jobs = [(stage, a, b, min(b+chunk_size, m))
                        for a in range(rank) for b in range(0, m, chunk_size)]
                shared = SharedCores(memory) if pool is not None else None
                try:
                    if shared is not None:
                        descriptor = shared.description
                        iterator = _bounded_ordered(pool, (j+(descriptor,) for j in jobs), 2*workers)
                    else:
                        iterator = (_compute(engine, slots, basis, memory, j) for j in jobs)
                    iterator = iter(iterator)
                    while True:
                        tw = perf_counter()
                        try:
                            job, responses, stats, cpu, elapsed = next(iterator)
                        except StopIteration:
                            break
                        wait += perf_counter()-tw
                        worker_sum += elapsed
                        if pool is not None:
                            _aggregate(engine, stats, cpu)
                        _, alpha, start, _ = job
                        ta = perf_counter()
                        for b, response in enumerate(responses, start=start):
                            if last:
                                final[alpha, b] = MPS(response).value([0]*len(response))
                            else:
                                tag = np.zeros((1, rank*m, 1), complex)
                                tag[0, alpha*m+b, 0] = 1
                                merger.push(MPS([tag, *response]))
                        assembly_time += perf_counter()-ta
                except BaseException:
                    # Pending tasks may not have attached the shared memory yet.
                    # Keep it alive until workers have stopped on a failure.
                    if pool is not None:
                        pool.shutdown(wait=True, cancel_futures=True)
                        pool = None
                    raise
                finally:
                    if shared is not None:
                        shared.__exit__()
                ts = perf_counter()
                if last:
                    core = _walsh_tag(final.reshape(1, rank*m, 1), rank, engine.n).reshape(rank, m, 1)
                    time_error = 0.
                else:
                    tagged = merger.finish()
                    tagged.cores[0] = _walsh_tag(tagged.cores[0], rank, engine.n)
                    canonical = tagged.right_canonical()
                    u, s, vh, time_error = split(canonical.cores[0].reshape(rank*m, -1),
                                                 time_truncation)
                    core = u.reshape(rank, m, len(s))
                    memory = [np.einsum("ij,jar->iar", s[:, None]*vh, canonical.cores[1]),
                              *canonical.cores[2:]]
                split_time = perf_counter()-ts
                out.append(core)
                record = dict(slot=stage, name=slot.name, probes=rank*m,
                    time_bond=core.shape[-1], time_discarded_norm=time_error,
                    assembly_discarded_norm_sum=merger.discarded,
                    workers=workers, blas_threads=blas_threads, chunk_size=chunk_size,
                    assembly=assembly, response_worker_seconds_sum=worker_sum,
                    response_wait_seconds=wait, assembly_seconds=assembly_time,
                    finish_and_split_seconds=split_time, elapsed_seconds=perf_counter()-t0)
                records.append(record)
                if progress is not None:
                    progress(record)
        return MPS(out), records
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def probability_report(p, *, max_entries=1_000_000, tolerance=1e-10):
    total = p.total()
    report = dict(total_real=float(total.real), total_imag=float(total.imag),
                  normalization_error=float(abs(total-1)),
                  entries=prod(p.dims), storage=p.storage, bonds=list(p.bonds),
                  positivity_checked=False)
    if prod(p.dims) <= max_entries:
        dense = p.dense(max_entries)
        report.update(positivity_checked=True,
                      min_probability=float(dense.real.min()),
                      negative_mass=float(-np.minimum(dense.real, 0).sum()),
                      max_imaginary=float(np.max(np.abs(dense.imag))))
        report["valid"] = (report["normalization_error"] <= tolerance
                           and report["min_probability"] >= -tolerance
                           and report["max_imaginary"] <= tolerance)
    else:
        report["valid"] = None  # normalization alone does NOT certify positivity
    return report


def marginals(p):
    right = [None]*(len(p.cores)+1)
    right[-1] = np.ones(1, complex)
    for j in range(len(p.cores)-1, -1, -1):
        right[j] = p.cores[j].sum(axis=1) @ right[j+1]
    left, out = np.ones(1, complex), []
    for j, c in enumerate(p.cores):
        out.append(np.einsum("l,lar,r->a", left, c, right[j+1]))
        left = left @ c.sum(axis=1)
    return out


def independent_model(p):
    return MPS.product(marginals(p))


def sample_histories(p, shots, rng, *, tolerance=1e-10, max_entries=1_000_000):
    """Exact conditionals up to roundoff, after exhaustive positivity check.

    SVD cores need not be entrywise positive. Until a positivity certificate is
    available we refuse unenumerable distributions, rather than silently sample
    only visited branches of a potentially signed tensor. Returns clipping audit.
    """
    if not isinstance(shots, int) or shots < 0:
        raise ValueError("shots must be a nonnegative integer")
    report = probability_report(p, max_entries=max_entries, tolerance=tolerance)
    if report["valid"] is not True:
        raise ValueError(f"sampling requires a verified probability tensor: {report}")
    right = [None]*(len(p.cores)+1)
    right[-1] = np.ones(1, complex)
    for j in range(len(p.cores)-1, -1, -1):
        right[j] = p.cores[j].sum(axis=1) @ right[j+1]
    histories = np.zeros((shots, len(p.cores)), int)
    clipped = 0.0
    for shot in range(shots):
        left = np.ones(1, complex)
        for j, core in enumerate(p.cores):
            probs = np.einsum("l,lar,r->a", left, core, right[j+1])
            denom = probs.sum()
            if abs(denom.imag) > tolerance or denom.real <= 0:
                raise ValueError("invalid conditioning denominator")
            probs = probs/denom
            if np.max(np.abs(probs.imag)) > tolerance or probs.real.min() < -tolerance:
                raise ValueError("negative or complex conditional probability")
            clipped += float(-np.minimum(probs.real, 0).sum())
            probs = np.maximum(probs.real, 0)
            probs /= probs.sum()
            label = rng.choice(len(probs), p=probs)
            histories[shot, j] = label
            left = left @ core[:, label, :]
            # Remove the joint-prefix magnitude to avoid long-chain underflow.
            left /= float((left @ right[j+1]).real)
    return histories, dict(clipped_conditional_mass_sum=clipped,
                           exhaustive_probability_report=report)
