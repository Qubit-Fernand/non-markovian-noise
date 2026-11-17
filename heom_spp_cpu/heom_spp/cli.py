"""Command line: build an SPP from layer/noise files, compare runs, summarize scaling."""
import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from .bath import NoiseSpec, build_noise
from .circuit import CircuitLayers, html_ising_layers, one_qubit_layers
from .heom import DenseHEOM, TDVPHEOM
from .mps import MPS, Truncation
from .native import ising_sqisw_layers
from .pulse import PulseDurations, compile_layers, system_terms
from .reference import static_reference_check
from .spp import available_cpus, build_spp, independent_model, marginals, probability_report


def _write(path, data):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _snapshot(output):
    root = Path(__file__).resolve().parents[1]
    hashes = {}
    for source in sorted((root/"heom_spp").glob("*.py")):
        target = output/"source"/"heom_spp"/source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(source.relative_to(root))] = sha256(source.read_bytes()).hexdigest()
    for folder in ("scripts", "tests"):
        for source in sorted((root/folder).glob("*")):
            if source.is_file() and source.suffix in (".py", ".sh", ".slurm"):
                target = output/"source"/folder/source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    for name in ("pyproject.toml", "README.md"):
        source = root/name
        if source.exists():
            target = output/"source"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    revision = None
    if (root/".git").exists():
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            patch = subprocess.check_output(["git", "diff", "--binary"], cwd=root)
            (output/"source"/"working-tree.patch").write_bytes(patch)
        except (OSError, subprocess.CalledProcessError):
            pass
    elif (root/"SOURCE_MANIFEST.json").exists():
        revision = json.loads((root/"SOURCE_MANIFEST.json").read_text(
            encoding="utf-8")).get("base_revision")
    return dict(upstream_head=revision, source_sha256=hashes)


def make_model(args):
    # Logical layer boundaries, pulse synthesis and bath specification are
    # independent inputs. Keep this function as a small orchestration layer.
    layers_file = getattr(args, "layers_json", None)
    if layers_file:
        plan = CircuitLayers.load(layers_file)
    elif args.example == "one-qubit":
        plan = one_qubit_layers(args.steps)
    elif args.example == "ising4-sqisw":
        plan = ising_sqisw_layers(4, steps=args.steps,
            theta_x=getattr(args, "theta_x", .5), theta_z=getattr(args, "theta_z", -.075))
    else:
        plan = html_ising_layers(4, steps=args.steps,
            theta_x=getattr(args, "theta_x", .5), theta_z=getattr(args, "theta_z", -.075))
    if args.slot_limit is not None:
        if not 1 <= args.slot_limit <= len(plan.layers):
            raise ValueError("slot-limit must select a nonempty circuit prefix")
        plan = CircuitLayers(plan.n_qubits, plan.layers[:args.slot_limit], plan.source)
    n = plan.n_qubits
    if getattr(args, "square_native", False):
        # Agreed native timing: 15 ns XY pulses, 20 ns sqisw, 5 ns idle before
        # every pulse block, every Z ideal.
        args.rotation_ns, args.sqisw_ns, args.buffer_ns, args.virtual_z = 15., 20., 5., True
    durations = PulseDurations(rotation_ns=args.rotation_ns, cx_ns=args.cx_ns,
        sqisw_ns=getattr(args, "sqisw_ns", 20.), buffer_ns=getattr(args, "buffer_ns", 0.),
        cz_ns=getattr(args, "cz_ns", 80.), rzz_ns=getattr(args, "rzz_ns", 80.),
        idle_ns=getattr(args, "idle_ns", 20.), virtual_z=getattr(args, "virtual_z", False))
    reset = getattr(args, "reset", "none")
    slots, schedule = compile_layers(plan, durations, reset_blocks=reset == "pulse")
    noise_file = getattr(args, "noise_json", None)
    if noise_file:
        spec = NoiseSpec.load(noise_file)
        if spec.n_qubits != n:
            raise ValueError("noise and circuit register sizes differ")
    else:
        coupled = None if args.coupled_qubits == "all" else tuple(
            int(q) for q in args.coupled_qubits.split(","))
        spec = NoiseSpec(n, kind=args.bath, sigma=args.sigma, levels=args.levels,
            topology=args.topology, coupled_qubits=coupled,
            modes_per_qubit=args.modes_per_qubit or (12 if args.bath == "one-over-f" else 1),
            gamma=args.gamma, gamma_min=args.gamma_min, gamma_max=args.gamma_max,
            hierarchy_basis=getattr(args, "hierarchy_basis", "scaled"),
            coupling=getattr(args, "noise_coupling", "Z/2"))
    noise = build_noise(spec)
    drift, jumps = system_terms(n, t1_ns=args.t1_ns, residual_zz=args.residual_zz)
    options = dict(max_dt=args.max_dt, truncation=Truncation(args.heom_bond, args.heom_cutoff),
                   jumps=jumps, drift=drift, hierarchy_basis=noise.hierarchy_basis)
    if args.engine == "dense":
        engine = DenseHEOM(noise.modes, max_entries=args.dense_max_entries, **options)
    else:
        engine = TDVPHEOM(noise.modes, integrator=args.engine,
            krylov_dim=args.krylov_dim, krylov_tol=args.krylov_tol, **options)
    return engine, slots, dict(n_qubits=n, bath_coordinates=len(noise.modes),
        convention="gate-after: R=V O", time_units="ns", frequency_units="rad/ns",
        hierarchy="one bath node per exponential term, levels per node",
        noise=noise.summary, circuit_plan=plan.to_dict(), pulse_schedule=schedule,
        bath_reset=reset, t1_ns=args.t1_ns, markovian_relaxation_on_all_ADOs=True,
        physical_twirl_pulses="ideal instantaneous",
        layers_input=str(layers_file) if layers_file else plan.source,
        noise_input=str(noise_file) if noise_file else "CLI NoiseSpec")


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k != "function"}
    _write(args.output/"config.json", config)
    try:
        engine, slots, model = make_model(args)
        _write(args.output/"layers.json", model["circuit_plan"])
        _write(args.output/"noise.json", model["noise"]["spec"])
        _write(args.output/"pulse_schedule.json", model["pulse_schedule"])
        source = _snapshot(args.output)
        versions = {}
        for name in ("numpy", "scipy", "threadpoolctl", "qiskit"):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                pass
        hardware = dict(platform=platform.platform(), python=platform.python_version(),
                        available_cpus=available_cpus(), logical_cpus=os.cpu_count(),
                        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)
        def progress(record):
            _write(args.output/f"stage_{record['slot']:03d}.json", record)
            print(json.dumps(record), flush=True)
        start = perf_counter()
        p, records = build_spp(engine, slots, workers=args.workers,
            blas_threads=args.blas_threads, chunk_size=args.chunk_size,
            assembly=args.assembly,
            time_truncation=Truncation(args.time_bond, args.time_cutoff),
            assembly_truncation=Truncation(args.assembly_bond, args.assembly_cutoff),
            progress=progress, reset=args.reset != "none")
        elapsed = perf_counter()-start
        p.save(args.output/"spp.npz")
        np.savez_compressed(args.output/"marginals.npz",
                           **{f"slot_{j}": x for j, x in enumerate(marginals(p))})
        report = dict(status="complete", config=config, model=model, source=source,
            hardware=hardware, versions=versions, construction=records,
            construction_seconds=elapsed, timing_scope="includes preparation, pool start/shutdown, propagation, assembly",
            hierarchy=asdict(engine.stats), tdvp=getattr(engine, "tdvp_stats", {}),
            probability=probability_report(p, max_entries=args.positivity_max_entries),
            threadpools=threadpool_info(),
            limitations=["noise.json presets are classical dephasing; complex baths use the Python API",
                "one-site TDVP rank and warm-start/time-step errors require convergence",
                "normalization does not certify unenumerated probability positivity",
                 "local discarded norms are not a global physical error bound",
                "raw/scaled hierarchy basis changes truncation norms; check convergence separately"])
        _write(args.output/"summary.json", report)
        print(f"construction_seconds={elapsed:.6f}; output={args.output}", flush=True)
    except BaseException as exc:
        _write(args.output/"failure.json", dict(type=type(exc).__name__, message=str(exc)))
        raise


def reference(args):
    """Quadrature SPP of static noise: the yardstick for the HEOM parameters."""
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k != "function"}
    _write(args.output/"config.json", config)
    try:
        engine, slots, model = make_model(args)
        _write(args.output/"layers.json", model["circuit_plan"])
        _write(args.output/"noise.json", model["noise"]["spec"])
        _write(args.output/"pulse_schedule.json", model["pulse_schedule"])
        p, info = static_reference_check(engine, slots, nodes=args.nodes,
            truncation=Truncation(args.time_bond, args.compression_cutoff))
        p.save(args.output/"spp.npz")
        correlation = p.add(independent_model(p).scaled(-1)).norm()
        _write(args.output/"summary.json", dict(status="complete", kind="reference",
            method="Gauss-Hermite quadrature over static detunings; no HEOM",
            config=config, model=model, quadrature=info, correlation_l2=correlation,
            construction_seconds=info["seconds"],
            probability=probability_report(p, max_entries=args.positivity_max_entries)))
        print(json.dumps(dict(info, correlation_l2=correlation,
                              normalization_error=abs(p.total()-1)), indent=2))
        print(f"output={args.output}", flush=True)
    except BaseException as exc:
        _write(args.output/"failure.json", dict(type=type(exc).__name__, message=str(exc)))
        raise


def compare(args):
    p, q = MPS.load(args.first/"spp.npz"), MPS.load(args.second/"spp.npz")
    if p.dims != q.dims:
        raise ValueError("different circuit/history dimensions")
    difference = p.add(q.scaled(-1))
    pm, qm = marginals(p), marginals(q)
    rng = np.random.default_rng(20261111)
    histories = [tuple(0 for _ in p.dims)]
    histories += [tuple(int(rng.integers(d)) for d in p.dims) for _ in range(128)]
    report = dict(first=str(args.first), second=str(args.second),
        joint_l2=difference.norm(),
        marginal_max_abs=max(float(np.max(abs(a-b))) for a, b in zip(pm, qm)),
        checked_history_max_abs=max(abs(p.value(a)-q.value(a)) for a in histories),
        probability_sum_difference=float(abs(p.total()-q.total())))
    print(json.dumps(report, indent=2))
    if args.output:
        _write(args.output, report)
    if args.tolerance is not None and max(report["joint_l2"], report["marginal_max_abs"]) > args.tolerance:
        raise SystemExit("comparison exceeds requested tolerance")


def summarize(args):
    from statistics import median
    rows, groups = [], {}
    for path in sorted(args.root.rglob("summary.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if "construction_seconds" not in d:
            continue
        c = d["config"]
        scenario = {k: v for k, v in c.items()
                    if k not in ("output", "workers")}
        fingerprint = json.dumps(scenario, sort_keys=True)
        row = dict(path=str(path.parent), engine=c["engine"], workers=c["workers"],
            threads=c["blas_threads"], chunk=c["chunk_size"],
            seconds=d["construction_seconds"],
            total_real=d["probability"]["total_real"],
            max_time_bond=max(x["time_bond"] for x in d["construction"]),
            assembly_seconds=sum(x["assembly_seconds"] for x in d["construction"]))
        rows.append(row)
        groups.setdefault(fingerprint, {}).setdefault(c["workers"], []).append(row["seconds"])
    scaling = []
    for fingerprint, by_workers in groups.items():
        base = median(by_workers[1]) if 1 in by_workers else None
        scenario = json.loads(fingerprint)
        for workers, values in sorted(by_workers.items()):
            elapsed = median(values)
            speedup = base/elapsed if base is not None else None
            scaling.append(dict(scenario=scenario, workers=workers, repeats=len(values),
                median_seconds=elapsed, min_seconds=min(values), max_seconds=max(values),
                speedup=speedup, efficiency=speedup/workers if speedup is not None else None))
    report = dict(runs=rows, scaling=scaling)
    print(json.dumps(report, indent=2))
    if args.output:
        _write(args.output, report)


# (config key, label, True if a SMALLER value is the finer setting)
_AXES = (("time_bond", "time bond", False), ("heom_bond", "HEOM bond", False),
         ("levels", "hierarchy levels", False), ("max_dt", "max dt (ns)", True))


def converge(args):
    """Pick, per parameter, the cheapest setting that agrees with the next finer one.

    Runs are the sub-directories of root written by scripts/scan_convergence.sh:
    `base` plus runs that differ from it in exactly one of the scanned
    parameters. Two neighbouring settings agree when the L2 distance of their
    joint distributions is below the tolerance, by default 1% of the
    correlation signal ||p - product of marginals|| of the base run.

    With a reference (root/reference from `cli reference`, or --reference),
    every run is instead measured against the reference directly and the
    fastest run within tolerance is recommended.
    """
    reference_dir = args.reference or (args.root/"reference" if (
        args.root/"reference"/"spp.npz").exists() else None)
    if reference_dir is not None:
        return _converge_to_reference(args, reference_dir)
    runs, unfinished = {}, []
    for folder in sorted(x for x in args.root.iterdir() if x.is_dir()):
        summary = folder/"summary.json"
        if not summary.exists():
            unfinished.append(folder.name)
            continue
        d = json.loads(summary.read_text(encoding="utf-8"))
        if d.get("kind") == "reference" or d["config"].get("reset", "none") != "none":
            continue
        setting = {k: d["config"][k] for k, _, _ in _AXES}
        setting["levels"] = d["model"]["noise"]["spec"]["levels"]
        runs[folder.name] = dict(setting=setting, seconds=d["construction_seconds"],
            normalization_error=d["probability"]["normalization_error"],
            bonds=d["probability"]["bonds"], path=folder)
    if "base" not in runs:
        raise SystemExit(f"{args.root}/base has no summary.json; unfinished: {unfinished}")
    load = lambda name: MPS.load(runs[name]["path"]/"spp.npz")
    base = runs["base"]
    p_base = load("base")
    signal = p_base.add(independent_model(p_base).scaled(-1)).norm()
    tolerance = args.tolerance if args.tolerance is not None else args.relative*signal
    lines = [f"scan {args.root}: {len(runs)} finished runs" + (f", UNFINISHED: {unfinished}" if unfinished else ""),
             f"base: " + ", ".join(f"{label} {base['setting'][k]}" for k, label, _ in _AXES)
             + f"  ({base['seconds']:.0f} s, time bonds {base['bonds']})",
             f"correlation signal ||p - product of marginals|| = {signal:.3e}",
             f"tolerance on ||p_a - p_b|| = {tolerance:.3e}", ""]
    recommended, report = {}, {}
    for key, label, smaller_is_finer in _AXES:
        ladder = sorted((name for name, r in runs.items() if all(
            r["setting"][k] == base["setting"][k] for k, _, _ in _AXES if k != key)),
            key=lambda name: runs[name]["setting"][key], reverse=smaller_is_finer)
        values = [runs[name]["setting"][key] for name in ladder]
        if len(ladder) < 2:
            lines.append(f"{label}: only {values[0]} was run, NOT CHECKED")
            recommended[key], report[key] = values[0], dict(values=values, status="not checked")
            continue
        steps, choice = [], None
        previous = load(ladder[0])
        for a, b in zip(ladder, ladder[1:]):
            current = load(b)
            distance = previous.add(current.scaled(-1)).norm()
            ok = distance <= tolerance
            steps.append(dict(coarse=runs[a]["setting"][key], fine=runs[b]["setting"][key],
                              joint_l2=distance, converged=bool(ok)))
            lines.append(f"{label}: {runs[a]['setting'][key]:>5} vs {runs[b]['setting'][key]:<5} "
                         f"||dp|| = {distance:.3e}  {'ok' if ok else 'not converged'}"
                         f"   ({runs[a]['seconds']:.0f} s vs {runs[b]['seconds']:.0f} s)")
            if ok and choice is None:
                choice = runs[a]["setting"][key]
            previous = current
        status = "converged" if choice is not None else "NOT CONVERGED: extend the scan"
        recommended[key] = choice if choice is not None else values[-1]
        report[key] = dict(values=values, steps=steps, status=status)
        lines.append(f"    -> {label} = {recommended[key]}  ({status})")
    exact = [name for name, r in runs.items() if r["setting"] == recommended]
    lines += ["", "recommended: " + ", ".join(f"{label} {recommended[k]}" for k, label, _ in _AXES),
              f"    --time-bond {recommended['time_bond']} --heom-bond {recommended['heom_bond']} "
              f"--levels {recommended['levels']} --max-dt {recommended['max_dt']}"]
    if exact:
        lines.append(f"    already computed: {runs[exact[0]]['path']}  "
                     f"(python -m heom_spp.analyze {runs[exact[0]]['path']})")
    else:
        lines.append("    this combination has not been run yet; the scan changed one parameter at a time")
    if any(v["status"] != "converged" for v in report.values()):
        lines.append("    WARNING: at least one parameter is unchecked or not converged")
    print("\n".join(lines))
    _write(args.output or args.root/"convergence.json", dict(root=str(args.root),
        correlation_signal=signal, tolerance=tolerance, base=base["setting"],
        recommended=recommended, axes=report, unfinished=unfinished,
        matching_run=str(runs[exact[0]]["path"]) if exact else None))


def _converge_to_reference(args, reference_dir):
    reference_dir = Path(reference_dir)
    q = MPS.load(reference_dir/"spp.npz")
    qm = marginals(q)
    signal = q.add(independent_model(q).scaled(-1)).norm()
    tolerance = args.tolerance if args.tolerance is not None else args.relative*signal
    rows, unfinished, skipped = [], [], []
    for folder in sorted(x for x in args.root.iterdir() if x.is_dir()):
        if folder.resolve() == reference_dir.resolve():
            continue
        if not (folder/"summary.json").exists():
            unfinished.append(folder.name)
            continue
        d = json.loads((folder/"summary.json").read_text(encoding="utf-8"))
        if d.get("kind") == "reference" or d["config"].get("reset", "none") != "none":
            skipped.append(folder.name)
            continue
        p = MPS.load(folder/"spp.npz")
        if p.dims != q.dims:
            skipped.append(folder.name)
            continue
        setting = {k: d["config"][k] for k, _, _ in _AXES}
        setting["levels"] = d["model"]["noise"]["spec"]["levels"]
        rows.append(dict(name=folder.name, path=str(folder), setting=setting,
            seconds=d["construction_seconds"], joint_l2=p.add(q.scaled(-1)).norm(),
            marginal_max_abs=max(float(np.max(abs(a-b))) for a, b in zip(marginals(p), qm)),
            correlation_l2=p.add(independent_model(p).scaled(-1)).norm()))
    if not rows:
        raise SystemExit(f"no finished HEOM runs under {args.root}; unfinished: {unfinished}")
    for row in rows:
        row["converged"] = bool(row["joint_l2"] <= tolerance)
    rows.sort(key=lambda r: r["seconds"])
    lines = [f"scan {args.root}: {len(rows)} finished HEOM runs against reference {reference_dir}"
             + (f", UNFINISHED: {unfinished}" if unfinished else ""),
             f"reference correlation signal ||p - product of marginals|| = {signal:.3e}",
             f"tolerance on ||p_run - p_reference|| = {tolerance:.3e}", "",
             f"{'run':12s}{'tb':>5s}{'hb':>5s}{'lv':>5s}{'dt':>7s}{'seconds':>10s}"
             f"{'||dp||':>12s}{'marg. max':>12s}{'corr. sig.':>12s}"]
    for r in rows:
        c = r["setting"]
        lines.append(f"{r['name']:12s}{c['time_bond']:5d}{c['heom_bond']:5d}{c['levels']:5d}"
                     f"{c['max_dt']:7g}{r['seconds']:10.0f}{r['joint_l2']:12.3e}"
                     f"{r['marginal_max_abs']:12.3e}{r['correlation_l2']:12.3e}"
                     f"  {'ok' if r['converged'] else ''}")
    good = [r for r in rows if r["converged"]]
    best = good[0] if good else min(rows, key=lambda r: r["joint_l2"])
    c = best["setting"]
    lines += ["", ("recommended (fastest run within tolerance): " if good else
                   "NO RUN WITHIN TOLERANCE; closest to the reference: ") + best["name"],
              f"    --time-bond {c['time_bond']} --heom-bond {c['heom_bond']} "
              f"--levels {c['levels']} --max-dt {c['max_dt']}",
              f"    python -m heom_spp.analyze {best['path']}"]
    print("\n".join(lines))
    _write(args.output or args.root/"convergence.json", dict(root=str(args.root),
        reference=str(reference_dir), correlation_signal=signal, tolerance=tolerance,
        runs=rows, recommended=best["setting"] if good else None,
        matching_run=best["path"] if good else None, closest_run=best["path"],
        unfinished=unfinished, skipped=skipped))


def _model_arguments(p):
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--example", choices=("one-qubit", "ising4", "ising4-sqisw"), default="one-qubit")
    p.add_argument("--layers-json", type=Path, help="explicit logical learning layers; overrides example/steps/angles")
    p.add_argument("--noise-json", type=Path, help="noise specification; overrides bath CLI settings")
    p.add_argument("--noise-coupling", choices=("Z/2", "Z"), default="Z/2")
    p.add_argument("--theta-x", type=float, default=.5)
    p.add_argument("--theta-z", type=float, default=-.075)
    p.add_argument("--hierarchy-basis", choices=("scaled", "unscaled"), default="scaled")
    p.add_argument("--bath", choices=("static", "ou", "one-over-f"), default="static")
    p.add_argument("--topology", choices=("independent", "shared"), default="independent")
    p.add_argument("--coupled-qubits", default="all")
    p.add_argument("--modes-per-qubit", type=int)
    p.add_argument("--levels", type=int, default=6)
    p.add_argument("--sigma", type=float, default=.004, help="total rms per coupled qubit, rad/ns")
    p.add_argument("--gamma", type=float, default=.01, help="OU decay, 1/ns")
    p.add_argument("--gamma-min", type=float, default=1e-4)
    p.add_argument("--gamma-max", type=float, default=.1)
    p.add_argument("--rotation-ns", type=float, default=20.)
    p.add_argument("--cx-ns", type=float, default=80.)
    p.add_argument("--cz-ns", type=float, default=80.)
    p.add_argument("--rzz-ns", type=float, default=80.)
    p.add_argument("--idle-ns", type=float, default=20.)
    p.add_argument("--virtual-z", action="store_true", help="rz/z as zero-duration frame updates")
    p.add_argument("--sqisw-ns", type=float, default=20.)
    p.add_argument("--buffer-ns", type=float, default=0., help="idle before every pulse block")
    p.add_argument("--square-native", action="store_true",
                   help="15 ns pulses, 20 ns sqisw, 5 ns buffers, ideal Z (overrides those options)")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--slot-limit", type=int)
    p.add_argument("--max-dt", type=float, default=2.)
    p.add_argument("--heom-bond", type=int, default=16)
    p.add_argument("--heom-cutoff", type=float, default=1e-13)
    p.add_argument("--time-cutoff", type=float, default=1e-12)
    p.add_argument("--krylov-dim", type=int, default=24)
    p.add_argument("--krylov-tol", type=float, default=1e-11)
    p.add_argument("--t1-ns", type=float, default=0.)
    p.add_argument("--residual-zz", type=float, default=0., help="nearest-neighbor ZZ coefficient, rad/ns")
    p.add_argument("--dense-max-entries", type=int, default=200000)
    p.add_argument("--positivity-max-entries", type=int, default=1000000)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("run")
    p.set_defaults(function=run)
    _model_arguments(p)
    p.add_argument("--engine", choices=("tdvp", "rk4", "dense"), default="tdvp")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--blas-threads", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--assembly", choices=("tree", "sequential"), default="tree")
    p.add_argument("--time-bond", type=int, default=16)
    p.add_argument("--assembly-bond", type=int, default=64)
    p.add_argument("--assembly-cutoff", type=float, default=1e-13)
    p.add_argument("--reset", choices=("none", "layer", "pulse"), default="none",
                   help="independent-noise baselines: zero the ADOs after every learning layer, "
                        "or after every pulse block as well")
    p = sub.add_parser("reference", help="exact SPP of static noise by quadrature (no HEOM)")
    p.set_defaults(function=reference, engine="tdvp")
    _model_arguments(p)
    p.add_argument("--nodes", type=int, default=6, help="Gauss-Hermite nodes per static mode")
    p.add_argument("--time-bond", type=int, default=128)
    p.add_argument("--compression-cutoff", type=float, default=1e-14)
    p = sub.add_parser("compare")
    p.set_defaults(function=compare)
    p.add_argument("first", type=Path)
    p.add_argument("second", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--tolerance", type=float)
    p = sub.add_parser("converge", help="choose settings from a scripts/scan_convergence.sh scan")
    p.set_defaults(function=converge)
    p.add_argument("root", type=Path)
    p.add_argument("--relative", type=float, default=.01,
                   help="tolerance as a fraction of the base run's correlation signal")
    p.add_argument("--tolerance", type=float, help="absolute tolerance on the joint L2 distance instead")
    p.add_argument("--reference", type=Path,
                   help="run directory of `cli reference`; default <root>/reference if present")
    p.add_argument("--output", type=Path)
    p = sub.add_parser("summarize")
    p.set_defaults(function=summarize)
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if hasattr(args, "blas_threads") and args.blas_threads < 1:
        parser.error("blas-threads must be positive")
    # Imports are already complete here; threadpoolctl also changes loaded BLAS.
    # Child-process environment is set before spawning in spp.build_spp.
    with threadpool_limits(limits=getattr(args, "blas_threads", 1)):
        args.function(args)


if __name__ == "__main__":
    main()
