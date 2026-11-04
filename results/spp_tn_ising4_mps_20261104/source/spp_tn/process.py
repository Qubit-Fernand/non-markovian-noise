"""Sequential HEOM-to-SPP construction without enumerating full histories."""
from math import prod

import numpy as np

from .mps import K, MPS, Truncation, split
from .pauli import characters, paulis


def build_spp(engine, slots, *, time_truncation=Truncation(),
              assembly_truncation=Truncation(), progress=None):
    """Retain a basis of reachable memory vectors at each time cut.

    Assembly rounding acts on tagged bath MPSs; time_truncation only acts at
    the tag/memory cut. Both are explicit and separate from engine truncation.
    At the last slot read the root directly, with no bath reset between slots.
    """
    slots = tuple(slots)
    if not slots:
        raise ValueError("at least one slot required")
    m = 4**engine.n
    ps, w = paulis(engine.n), characters(engine.n)/m
    memory_cores = engine.root().cores
    cores, records = [], []
    for j, slot in enumerate(slots):
        rank = memory_cores[0].shape[0]
        last = j == len(slots)-1
        final = np.zeros((rank, m), complex) if last else None
        tagged, assembly_errors = None, []
        for alpha in range(rank):
            memory = MPS([memory_cores[0][alpha:alpha+1], *memory_cores[1:]])
            for b, p in enumerate(ps):
                response = engine.response(slot, p, memory)
                if last:
                    final[alpha, b] = response.value([0]*len(response.cores))
                else:
                    tag = np.zeros((1, rank*m, 1), complex)
                    tag[0, alpha*m+b, 0] = 1
                    branch = MPS([tag, *response.cores])
                    if tagged is None:
                        tagged = branch
                    else:
                        tagged, errors = tagged.add(branch).round(assembly_truncation)
                        assembly_errors.extend(errors)
        if last:
            core = (final @ w.T).reshape(rank, m, 1)
            time_errors = []
        else:
            # Walsh transform on b without mixing the previous time bond alpha.
            tag = tagged.cores[0].reshape(rank, m, -1)
            tagged.cores[0] = K.einsum("ab,xbr->xar", w, tag).reshape(1, rank*m, -1)
            # Right-canonicalize bath before splitting: local SVD sees its norm.
            canonical = tagged.right_canonical()
            u, s, vh, err = split(canonical.cores[0].reshape(rank*m, -1), time_truncation)
            core = u.reshape(rank, m, len(s))
            memory_cores = [K.einsum("ij,jar->iar", s[:, None]*vh, canonical.cores[1]),
                            *canonical.cores[2:]]
            time_errors = [err]
        cores.append(core)
        records.append(dict(slot=j, name=slot.name, probes=rank*m,
                            time_bond=core.shape[2],
                            time_discarded_norm=sum(time_errors),
                            assembly_discarded_norm_sum=sum(assembly_errors)))
        if progress is not None:
            progress(records[-1])
    return MPS(cores), records


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
        out.append(K.einsum("l,lar,r->a", left, c, right[j+1]))
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
            probs = K.einsum("l,lar,r->a", left, core, right[j+1])
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
