#!/usr/bin/env python3
"""Run the reproducible three-qubit classical and one-qubit quantum benchmarks."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from nmnoise.core import ghz_schedule, ideal_state, ensemble_rho, state_summary, trace_distance
from nmnoise.classical import ou_noise, dense_trajectories, lindblad, ou_chi
from nmnoise.mps import mps_trajectories


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qubits", type=int, default=3, choices=[2,3,4])
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--dt", type=float, default=2.)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--sigma", type=float, default=0.004)
    ap.add_argument("--skip-quantum", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("results/demo"))
    args = ap.parse_args()
    out = args.output
    if (out/"summary.json").exists():
        ap.error("Output already contains results; choose another --output directory")
    out.mkdir(parents=True, exist_ok=True)
    pulses = ghz_schedule(args.qubits)
    target = ideal_state(pulses, args.qubits)
    report = {"config": vars(args).copy(), "models": {}, "checks": {}}
    report["config"]["output"] = str(out)
    report["config"]["duration_ns"] = sum(p.duration for p in pulses)
    report["config"]["control"] = [dict(label=p.label, duration_ns=p.duration, sites=p.sites) for p in pulses]
    report["config"]["ou_modes"] = 12
    report["config"]["ou_rates_per_ns"] = [1e-4, 0.1]
    rhos = {}

    def record(name, rho, seconds, states=None):
        rhos[name] = rho
        entry = state_summary(rho, target)
        entry["seconds"] = seconds
        if states is not None and len(states) > 1:
            fidelities = np.abs(states @ target.conj())**2
            entry["fidelity_mc_standard_error"] = float(fidelities.std(ddof=1)/np.sqrt(len(states)))
        report["models"][name] = entry
        print(name, "F=", round(entry["fidelity"], 7), "time=", round(seconds,2), flush=True)

    record("ideal", np.outer(target, target.conj()), 0.)
    noise = ou_noise(pulses, args.qubits, args.dt, args.samples, sigma=args.sigma, seed=args.seed)
    # Persist the exact noise used by BOTH numerical representations.
    np.save(out/"colored_noise.npy", noise)
    start = time.perf_counter()
    dense = dense_trajectories(pulses, noise, args.dt)
    record("colored_dense", ensemble_rho(dense), time.perf_counter()-start, dense)
    for substeps in (1, 2):
        start = time.perf_counter()
        states, diagnostics = mps_trajectories(pulses, noise, args.dt, substeps=substeps)
        name = f"colored_mps_substeps{substeps}"
        record(name, ensemble_rho(states), time.perf_counter()-start, states)
        report["models"][name]["mps"] = diagnostics
        report["checks"][name+"_trace_distance_to_dense"] = trace_distance(ensemble_rho(states), rhos["colored_dense"])
    for name, kwargs in [("quasistatic_dense", dict(static=True)), ("reset_colored_dense", dict(reset=True))]:
        start = time.perf_counter()
        values = ou_noise(pulses, args.qubits, args.dt, args.samples, sigma=args.sigma, seed=args.seed, **kwargs)
        states = dense_trajectories(pulses, values, args.dt)
        record(name, ensemble_rho(states), time.perf_counter()-start, states)
    # Match ONE Ramsey calibration point, not an unknown circuit output.
    calibration_time = 100.
    gamma = ou_chi(calibration_time, sigma=args.sigma)/calibration_time
    report["config"]["markovian_calibration"] = dict(ramsey_time_ns=calibration_time, gamma_phi_per_ns=gamma)
    start = time.perf_counter()
    record("lindblad", lindblad(pulses, args.qubits, gamma), time.perf_counter()-start)
    report["checks"]["continuous_vs_reset_trace_distance"] = trace_distance(rhos["colored_dense"], rhos["reset_colored_dense"])
    report["checks"]["memory_comparison_note"] = "Finite Monte Carlo difference; not a significance test or hardware result."
    # Coupled grid check: average adjacent fine cells to preserve their total
    # stochastic phase. Independent paths would obscure integrator error with
    # Monte Carlo fluctuations. This tests time ordering within paired cells;
    # continuous-noise accuracy also has the analytic Ramsey test in pytest.
    start = time.perf_counter()
    fine = ou_noise(pulses, args.qubits, args.dt/2, args.samples, sigma=args.sigma, seed=args.seed+1)
    fine_states = dense_trajectories(pulses, fine, args.dt/2)
    record("colored_dense_half_dt", ensemble_rho(fine_states), time.perf_counter()-start, fine_states)
    averaged = fine.reshape(args.samples, -1, 2, args.qubits).mean(axis=2)
    averaged_states = dense_trajectories(pulses, averaged, args.dt)
    differences = np.abs(fine_states @ target.conj())**2 - np.abs(averaged_states @ target.conj())**2
    report["checks"]["paired_grid_refinement"] = dict(
        trace_distance=trace_distance(ensemble_rho(fine_states),ensemble_rho(averaged_states)),
        fidelity_difference=float(differences.mean()),
        fidelity_difference_standard_error=float(differences.std(ddof=1)/np.sqrt(args.samples)) if args.samples>1 else None,
        note="Paired fine cells versus their coarse average; probes control/noise time ordering, not full continuum convergence.")

    if not args.skip_quantum:
        from nmnoise.quantum import heom, explicit_bath, quantum_schedule
        qp = quantum_schedule()
        qtarget = ideal_state(qp, 1)
        quantum = {"description": "One qubit, two thermal oscillator modes: log quadrature of regularized 1/f symmetrized spectrum; illustrative, not continuum-converged or device-calibrated.",
                   "config": dict(modes=2, omega_min=0.015, omega_max=0.24, sigma=0.03, beta=60., duration_ns=40.),
                   "models": {}}
        qstates = {}
        for name, fn, kw in [("heom_depth4", heom, dict(depth=4)),
                              ("heom_depth6", heom, dict(depth=6)),
                              ("heom_reset_depth6", heom, dict(depth=6, reset=True)),
                              ("explicit_fock5", explicit_bath, dict(fock=5)),
                              ("explicit_fock7", explicit_bath, dict(fock=7))]:
            start = time.perf_counter()
            evolution, diag = fn(**kw)
            qstates[name] = evolution
            quantum["models"][name] = dict(state_summary(evolution[-1], qtarget), seconds=time.perf_counter()-start, diagnostics=diag)
            print(name, "rho=", evolution[-1].round(6).tolist(), flush=True)
        quantum["checks"] = {
            "heom_depth4_vs6": trace_distance(qstates["heom_depth4"][-1], qstates["heom_depth6"][-1]),
            "fock5_vs7": trace_distance(qstates["explicit_fock5"][-1], qstates["explicit_fock7"][-1]),
            "heom6_vs_explicit7": trace_distance(qstates["heom_depth6"][-1], qstates["explicit_fock7"][-1]),
            "heom_continuous_vs_reset": trace_distance(qstates["heom_depth6"][-1], qstates["heom_reset_depth6"][-1]),
        }
        report["quantum"] = quantum
        np.savez(out/"quantum_states.npz", **qstates)
    np.savez(out/"final_states.npz", **rhos)
    report["versions"] = {"numpy": np.__version__}
    import scipy, qutip
    report["versions"].update(scipy=scipy.__version__, qutip=qutip.__version__)
    (out/"summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    make_plot(out, rhos, report)
    print("Saved", out.resolve(), flush=True)


def make_plot(out, rhos, report):
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 16, "text.usetex": True})
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    keys = ["ideal", "lindblad", "quasistatic_dense", "colored_dense", "colored_mps_substeps2", "reset_colored_dense"]
    labels = ["Ideal", "Lindblad", "Static", "Colored", "MPS", "Reset"]
    values = [report["models"][k]["fidelity"] for k in keys]
    errors = [report["models"][k].get("fidelity_mc_standard_error", 0) for k in keys]
    axes[0].bar(labels, values, yerr=errors, capsize=3, color=["#aaaaaa", "#8ca3b8", "#d9b780", "#296a91", "#52a5a0", "#c47a75"])
    axes[0].set_ylim(0, 1.06)
    axes[0].set_ylabel("Fidelity to ideal GHZ")
    axes[0].tick_params(axis="x", rotation=35)
    im = axes[1].imshow(rhos["colored_dense"].real, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[1].set_title(r"Colored-noise $\mathrm{Re}(\rho)$")
    axes[1].set_xlabel("Computational basis index")
    axes[1].set_ylabel("Computational basis index")
    fig.colorbar(im, ax=axes[1], shrink=.8)
    z = np.load(out/"colored_noise.npy", mmap_mode="r")
    t = (np.arange(z.shape[1])+.5)*report["config"]["dt"]
    for q in range(z.shape[2]):
        axes[2].plot(t, z[0,:,q], label=rf"$q_{q}$", lw=1.3)
    axes[2].set_xlabel(r"Time (ns)")
    axes[2].set_ylabel(r"Detuning (rad/ns)")
    axes[2].set_title("One continuous noise trajectory")
    axes[2].legend(frameon=False)
    fig.savefig(out/"comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
