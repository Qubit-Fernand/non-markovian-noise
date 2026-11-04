"""Reproducible synthetic benchmark: python -m spp_tn.demo --output PATH."""
import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib import metadata
from pathlib import Path
import json
import platform
import shutil
import time

import numpy as np

from .circuits import embed, ising_slots, response_tensor
from .heom import DenseHermiteHEOM, HermiteHEOM, Mode, Segment, Slot
from .mitigation import (amplify, inverse_als, inverse_dense, quasiprobability_report,
                         sample_corrections)
from .mps import MPS, Truncation
from .pauli import X, Z, walsh
from .process import build_spp, independent_model, probability_report


def versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy", "tensorcircuit", "tensorcircuit-ng", "tensornetwork", "tensornetwork-ng"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def source_hashes():
    package = Path(__file__).parent
    return {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob("*.py"))}


def snapshot_sources(output):
    package = Path(__file__).parent
    target = output/"source"
    files = [*package.glob("*.py"), package/"README.md",
             package.parent/"pyproject.toml", package.parent/"tests"/"test_spp_tn.py"]
    hashes = {}
    for source in files:
        if source.is_file():
            relative = source.relative_to(package.parent)
            destination = target/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            hashes[str(relative)] = sha256(source.read_bytes()).hexdigest()
    return hashes


def convolve(p, q, n):
    return walsh(walsh(p, n).hadamard(walsh(q, n)), n, inverse=True)


def reference_convergence(slots, modes):
    """One-qubit small-system convergence evidence, with separate controls."""
    base = dict(operator=modes[0].operator, sigma=modes[0].sigma, gamma=modes[0].gamma)
    reference, _ = build_spp(DenseHermiteHEOM([Mode(**base, levels=10)]), slots,
                             time_truncation=Truncation(cutoff=1e-13))
    dense_ref = reference.dense()
    records = []
    for levels in (3, 5, 7):
        p, _ = build_spp(DenseHermiteHEOM([Mode(**base, levels=levels)]), slots,
                         time_truncation=Truncation(cutoff=1e-13))
        records.append(dict(axis="hierarchy_levels", value=levels,
                            error_l2=float(np.linalg.norm(p.dense()-dense_ref))))
    fixed_modes = [Mode(**base, levels=5)]
    fixed, _ = build_spp(DenseHermiteHEOM(fixed_modes), slots,
                         time_truncation=Truncation(cutoff=1e-13))
    for dt in (.2, .1, .05):
        engine = HermiteHEOM(fixed_modes, max_dt=dt, truncation=Truncation(cutoff=1e-13))
        p, _ = build_spp(engine, slots, time_truncation=Truncation(cutoff=1e-13))
        records.append(dict(axis="rk4_max_dt", value=dt,
                            error_l2=float(np.linalg.norm(p.dense()-fixed.dense()))))
    for bond in (1, 2, 4):
        engine = HermiteHEOM(fixed_modes, max_dt=.05,
                             truncation=Truncation(max_bond=bond, cutoff=1e-13))
        p, _ = build_spp(engine, slots, time_truncation=Truncation(cutoff=1e-13))
        records.append(dict(axis="internal_heom_bond", value=bond,
                            error_l2=float(np.linalg.norm(p.dense()-fixed.dense())),
                            probability=probability_report(p)))
        timed, _ = build_spp(DenseHermiteHEOM(fixed_modes), slots,
                             time_truncation=Truncation(max_bond=bond, cutoff=1e-13))
        records.append(dict(axis="time_bond", value=bond,
                            error_l2=float(np.linalg.norm(timed.dense()-fixed.dense())),
                            probability=probability_report(timed)))
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory; never overwrite")
    parser.add_argument("--example", choices=("one-qubit", "ising4"), default="one-qubit")
    parser.add_argument("--engine", choices=("mps", "dense"), default="mps")
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--max-dt", type=float, default=.05)
    parser.add_argument("--heom-bond", type=int, default=16)
    parser.add_argument("--time-bond", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=.55)
    parser.add_argument("--gamma", type=float, default=.2)
    parser.add_argument("--shots", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--convergence", action="store_true")
    args = parser.parse_args(argv)
    if args.shots < 1:
        parser.error("shots must be positive")
    if args.convergence and args.example != "one-qubit":
        parser.error("convergence sweep is defined for the one-qubit benchmark")
    n = 1 if args.example == "one-qubit" else 4
    if n == 1:
        slots = (Slot((Segment(.3*X, .7),), "rx"),
                 Slot((Segment(.2*Z, .7),), "rz"),
                 Slot((Segment(-.2*X, .7),), "rx-minus"))
        modes = [Mode(Z/2, args.sigma, args.gamma, args.levels)]
    else:
        slots = ising_slots()
        # Explicitly one mode on qubit zero; users can supply independent modes
        # on all qubits through the Python API. Do not silently claim eight modes.
        modes = [Mode(embed(Z, (0,), 4)/2, args.sigma, args.gamma, args.levels)]
    engine_type = HermiteHEOM if args.engine == "mps" else DenseHermiteHEOM
    engine = engine_type(modes, max_dt=args.max_dt,
                         truncation=Truncation(args.heom_bond, 1e-13))
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot_hashes = snapshot_sources(args.output)
    config = vars(args).copy()
    config["output"] = str(args.output)
    config.update(n_qubits=n, modes="one independent classical OU mode on qubit 0",
                  convention="gate-after", pauli_order="I,X,Y,Z; qubit 0 most significant",
                  time_units="arbitrary, sigma and gamma use reciprocal units",
                  slot_names=[s.name for s in slots],
                  slot_durations=[[s.duration for s in slot.segments] for slot in slots])
    (args.output/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    start = time.perf_counter()
    p, construction = build_spp(engine, slots,
                                time_truncation=Truncation(args.time_bond, 1e-12),
                                assembly_truncation=Truncation(cutoff=1e-13),
                                progress=lambda r: print(json.dumps(r), flush=True))
    p.save(args.output/"spp.npz")
    summary = dict(status="synthetic simulation", config=config, versions=versions(),
                   source_sha256=source_hashes(), snapshot_sha256=snapshot_hashes,
                   probability=probability_report(p, tolerance=1e-8),
                   construction=construction, hierarchy=asdict(engine.stats))
    if n == 1:
        dense, _ = build_spp(DenseHermiteHEOM(modes), slots,
                             time_truncation=Truncation(cutoff=1e-13))
        dense.save(args.output/"spp_dense_reference.npz")
        summary["dense_reference_error_l2"] = float(np.linalg.norm(p.dense()-dense.dense()))
        q, optimization = inverse_als(p, 1, bond=4, sweeps=20, tolerance=1e-7, seed=args.seed)
        q_exact = inverse_dense(p, 1)
        q.save(args.output/"pec.npz")
        summary["pec"] = dict(optimization=optimization, **quasiprobability_report(q),
                              error_to_dense_inverse_l2=float(np.linalg.norm(q.dense()-q_exact.dense())))
        response = response_tensor(slots, X, np.ones(2)/np.sqrt(2))
        expectation = lambda distribution: float(np.sum(distribution.dense()*response).real)
        independent = independent_model(p)
        q_independent = inverse_dense(independent, 1)
        gains = []
        for gain in range(4):
            amplified, audit = amplify(p, 1, gain, Truncation(cutoff=1e-13))
            gains.append(dict(gain=gain, expectation=expectation(amplified),
                              probability=probability_report(amplified), truncation=audit))
        summary["observables"] = dict(ideal=float(response[(0,)*len(slots)]),
                                      noisy=expectation(p), independent_prediction=expectation(independent),
                                      correlated_pec=expectation(convolve(p, q, 1)),
                                      independent_pec=expectation(convolve(p, q_independent, 1)),
                                      pea=gains,
                                      pea_quadratic_extrapolation=3*gains[1]["expectation"]
                                      -3*gains[2]["expectation"]+gains[3]["expectation"])
        # Finite-shot PEC: separately sample a correction, a physical error,
        # and the +/-1 Pauli measurement. Model knowledge is exact here.
        if summary["probability"]["valid"] is not True:
            raise ValueError("SPP failed probability checks; refusing finite-shot PEC")
        rng = np.random.default_rng(args.seed)
        corrections, weights, _ = sample_corrections(q, args.shots, rng)
        probabilities = p.dense().real.reshape(-1)
        clipped_mass = float(-np.minimum(probabilities, 0).sum())
        probabilities = np.maximum(probabilities, 0)
        probabilities /= probabilities.sum()
        indices = rng.choice(len(probabilities), size=args.shots, p=probabilities)
        noise = np.array(np.unravel_index(indices, p.dims)).T
        effective = noise ^ corrections
        means = response[tuple(effective.T)]
        outcomes = np.where(rng.random(args.shots) < (1+means)/2, 1, -1)
        estimator = weights*outcomes
        summary["finite_shot_pec"] = dict(estimate=float(estimator.mean()),
                                         standard_error=float(estimator.std(ddof=1)/np.sqrt(args.shots))
                                         if args.shots > 1 else None,
                                         shots=args.shots, negative_mass_clipped=clipped_mass,
                                         includes_learning_uncertainty=False)
        with open(args.output/"samples.npz", "xb") as stream:
            np.savez_compressed(stream, noise=noise, corrections=corrections,
                                weights=weights, outcomes=outcomes, response=response)
        if args.convergence:
            summary["convergence"] = reference_convergence(slots, modes)
    else:
        summary["limitations"] = ["full history positivity not enumerated (256^4 entries)",
                                   "PEA/PEC sampling and dense circuit response skipped at this size",
                                   "one bath mode, not the full per-qubit multimode target"]
    summary["elapsed_seconds"] = time.perf_counter()-start
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(f"Saved {args.output/'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
