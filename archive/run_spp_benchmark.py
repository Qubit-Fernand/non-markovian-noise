#!/usr/bin/env python3
"""Reproducible bounded HEOM -> two-slot SPP -> learning -> history PEC."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import numpy as np
from nmnoise.core import trace_distance
from nmnoise.spp import (ProcessConfig, CALIBRATION, heom_probabilities,
    explicit_probabilities, checked_probability, independent_projection,
    fit_terminal_counts, inverse_qp, convolution_matrix, ising_circuits,
    history_states, response_matrix, twirled_heom_state, pec_statistics)


def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def run(args):
    if args.train_shots < 100 or args.pec_shots < 100 or args.repeats < 20:
        raise ValueError('Require >=100 shots and >=20 learning repetitions')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    root = Path(__file__).resolve().parent
    sources = ['run_spp_benchmark.py', 'nmnoise/spp.py', 'nmnoise/quantum.py',
               'nmnoise/core.py', 'tests/test_spp.py', 'pyproject.toml']
    hashes = {}
    for name in sources:
        dest = out/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, dest)
        hashes[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    (out/'git-status.txt').write_text(subprocess.check_output(
        ['git', 'status', '--short'], cwd=root, text=True))
    (out/'working-tree.patch').write_text(subprocess.check_output(
        ['git', 'diff', '--', *sources], cwd=root, text=True))
    config = ProcessConfig()
    convergence = {'heom': [], 'explicit_bath': []}
    raw_heom, raw_fock = [], []
    for depth in (4, 6, 8, 10):
        p, info = heom_probabilities(config, depth)
        raw_heom.append(p)
        convergence['heom'].append(dict(**info, probabilities=p.tolist()))
        print(f'HEOM depth {depth}: {p}', flush=True)
    for fock in (4, 6, 8, 10, 12):
        p, info = explicit_probabilities(config, fock)
        raw_fock.append(p)
        convergence['explicit_bath'].append(dict(**info, probabilities=p.tolist()))
    truth = checked_probability(raw_heom[-1])
    for row in convergence['heom']+convergence['explicit_bath']:
        row['max_abs_vs_fock12'] = float(np.max(np.abs(np.array(row['probabilities'])-raw_fock[-1])))
    tight, _ = heom_probabilities(config, depth=10, tolerance=1e-12)
    convergence['depth8_to_10_max_abs'] = float(np.max(np.abs(raw_heom[-2]-truth)))
    convergence['fock10_to_12_max_abs'] = float(np.max(np.abs(raw_fock[-2]-raw_fock[-1])))
    convergence['tighter_ode_max_abs'] = float(np.max(np.abs(tight-truth)))
    convergence['continuum_spectral_convergence'] = 'not tested; fixed single oscillator model'
    if max(convergence['depth8_to_10_max_abs'], convergence['fock10_to_12_max_abs'],
           convergence['tighter_ode_max_abs'], np.max(np.abs(truth-raw_fock[-1]))) > 1e-6:
        raise RuntimeError('Physical generator failed convergence gate')

    # Seeds allocated before observing held-out results. No model selection on test.
    seeds = dict(calibration=args.seed, circuits=args.seed+1, validation=args.seed+2,
                 pec=args.seed+3, repeated_learning=args.seed+4)
    rng = np.random.default_rng(seeds['calibration'])
    minus_prob = (1-CALIBRATION @ truth)/2
    counts = rng.binomial(args.train_shots, minus_prob)
    np.savez_compressed(out/'training.npz', response=CALIBRATION,
                        minus_counts=counts, shots=np.full(3, args.train_shots))
    # Reload the saved, terminal-only dataset: fitting has no latent truth input.
    with np.load(out/'training.npz') as training:
        correlated, corr_fit = fit_terminal_counts(training['minus_counts'], args.train_shots)
        independent, ind_fit = fit_terminal_counts(training['minus_counts'], args.train_shots, True)
    models = dict(independent=independent, correlated=correlated,
                  oracle_independent=independent_projection(truth), oracle_correlated=truth)
    qps, inverse_info = {}, {}
    for name, p in models.items():
        qps[name], inverse_info[name] = inverse_qp(p)
    qps['unmitigated'] = np.array([1., 0, 0, 0])
    inverse_info['unmitigated'] = dict(gamma=1., gamma_squared=1.)
    circuits, observables, params = ising_circuits(seeds['circuits'], 12)
    response = response_matrix(circuits, observables)
    actual = response @ truth
    ideal = response[:, :, 0]
    validation_rng = np.random.default_rng(seeds['validation'])
    validation_counts = validation_rng.binomial(args.train_shots, (1-actual)/2)
    observed = 1-2*validation_counts/args.train_shots
    predictions = {}
    for name, p in models.items():
        delta = response @ p-actual
        predictions[name] = dict(expectation_rmse=rmse(delta),
                                 expectation_max_abs=float(np.abs(delta).max()),
                                 binary_probability_rmse=rmse(delta)/2,
                                 heldout_count_expectation_rmse=rmse(response @ p-observed))

    # Direct system+ADO integration checks both the twirl and correction interface.
    direct_checks, direct_states = [], []
    for ci in (0, 1, 2):
        states = history_states(circuits[ci])
        for correction in (range(4) if ci == 0 else [0]):
            direct = twirled_heom_state(circuits[ci], config, depth=10, correction=correction)
            predicted = np.einsum('h,hij->ij', truth[np.arange(4)^correction], states)
            distance = trace_distance(direct, predicted)
            direct_checks.append(dict(circuit=ci, correction=correction,
                                      trace_distance=distance,
                                      trace_error=float(abs(np.trace(direct)-1)),
                                      min_eigenvalue=float(np.linalg.eigvalsh(direct).min())))
            direct_states.append(direct)
            if distance > 1e-6:
                raise RuntimeError('Direct HEOM twirl/correction bridge failed')
    raw_state = twirled_heom_state(circuits[0], config, depth=10, twirl=False)
    raw_vs_twirl = trace_distance(raw_state, direct_states[0])
    print('Direct four-qubit twirl and all four corrections verified.', flush=True)

    pec_rng = np.random.default_rng(seeds['pec'])
    pec, records, count_archive = {}, [], {}
    obs_names = ['X0', 'X1', 'X2', 'X3', 'Z0Z1', 'Z1Z2', 'Z2Z3']
    for name, q in qps.items():
        estimates, means, variances, raw_counts = [], [], [], []
        for ci in range(len(circuits)):
            for oi, label in enumerate(obs_names):
                est, mean, variance, ct = pec_statistics(response[ci, oi], truth, q,
                                                       args.pec_shots, pec_rng)
                estimates.append(est); means.append(mean); variances.append(variance); raw_counts.append(ct)
                records.append(dict(model=name, circuit=ci, observable=label,
                                    ideal=ideal[ci, oi], noisy=actual[ci, oi],
                                    estimate=est, exact_mitigated=mean,
                                    conditional_standard_error=np.sqrt(variance)))
        bias = np.array(means)-ideal.ravel()
        pec[name] = dict(exact_bias_rmse=rmse(bias),
                         sampled_rmse=rmse(np.array(estimates)-ideal.ravel()),
                         mean_conditional_variance=float(np.mean(variances)),
                         conditional_expected_mse=float(np.mean(bias**2+variances)),
                         **inverse_info[name])
        count_archive[name] = np.array(raw_counts).reshape(12, 7, 4, 2)
    with (out/'pec_observables.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

    # Repeated independent training sets, not a test-driven tuning procedure.
    # Exact conditional MSE integrates PEC shot randomness; also draw actual PEC
    # estimates to check repeated end-to-end MSE and conditional SE calibration.
    learning_rng = np.random.default_rng(seeds['repeated_learning'])
    repeated = {name: [] for name in ('independent', 'correlated')}
    repeated_p, repeated_counts = [], []
    for _ in range(args.repeats):
        ct = learning_rng.binomial(args.train_shots, minus_prob)
        repeated_counts.append(ct)
        ps = []
        for name in repeated:
            p, _ = fit_terminal_counts(ct, args.train_shots, name == 'independent')
            q, info = inverse_qp(p)
            ps.append(p)
            mean = response @ (convolution_matrix(truth) @ q)
            variance = (info['gamma_squared']-mean**2)/args.pec_shots
            draws = np.array([pec_statistics(row, truth, q, args.pec_shots, learning_rng)[0]
                              for row in response.reshape(-1, 4)]).reshape(12, 7)
            repeated[name].append([rmse(response @ p-actual),
                float(np.mean((mean-ideal)**2+variance)),
                float(np.mean((draws-ideal)**2)), info['gamma_squared'],
                float(np.mean(np.abs(draws-mean) <= 1.96*np.sqrt(variance)))])
        repeated_p.append(ps)
    repeated_summary = {}
    for name, values in repeated.items():
        values = np.asarray(values)
        repeated_summary[name] = dict(
            prediction_rmse_median=float(np.median(values[:, 0])),
            prediction_rmse_q025_q975=np.quantile(values[:, 0], [.025, .975]).tolist(),
            total_expected_mse=float(np.mean(values[:, 1])),
            total_expected_mse_repetition_se=float(values[:, 1].std(ddof=1)/np.sqrt(args.repeats)),
            empirical_end_to_end_mse=float(np.mean(values[:, 2])),
            gamma_squared_q025_q975=np.quantile(values[:, 3], [.025, .975]).tolist(),
            conditional_95pct_coverage=float(np.mean(values[:, 4])))

    np.savez_compressed(out/'audit.npz', truth_raw=np.array(raw_heom),
        explicit_raw=np.array(raw_fock), tight_ode=tight, truth=truth,
        response=response, ideal=ideal, heldout_true=actual,
        validation_minus_counts=validation_counts, direct_states=np.array(direct_states),
        untwirled_state=raw_state, controls=np.asarray(circuits), observables=observables,
        repeated_fit_p=np.array(repeated_p), repeated_training_counts=np.array(repeated_counts),
        **{f'repeated_{k}': np.array(v) for k, v in repeated.items()},
        **{f'pec_counts_{k}': v for k, v in count_archive.items()})
    summary = dict(
        schema_version=1, evidence='synthetic numerical benchmark; no hardware data',
        config=asdict(config), seeds=seeds, history_order=['00', '01', '10', '11'],
        versions={k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'qutip')},
        python=platform.python_version(), source_sha256=hashes,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        convergence=convergence, direct_heom_checks=direct_checks,
        untwirled_to_twirled_trace_distance=raw_vs_twirl,
        truth=truth.tolist(), truth_temporal_covariance=float(truth[3]-truth[2:].sum()*truth[1::2].sum()),
        training=dict(shots_per_setting=args.train_shots, settings=3,
                      total_shots=3*args.train_shots, minus_counts=counts.tolist(),
                      augmented_response_rank=int(np.linalg.matrix_rank(np.vstack([np.ones(4), CALIBRATION]))),
                      augmented_response_condition=float(np.linalg.cond(np.vstack([np.ones(4), CALIBRATION]))),
                      fits=dict(independent=ind_fit, correlated=corr_fit)),
        models={k: v.tolist() for k, v in models.items()},
        inverse_coefficients={k: v.tolist() for k, v in qps.items()},
        ising=dict(qubits=4, layers=2, heldout_circuits=12, parameters_t_J_h=params,
                   observable_names=obs_names, physical_control_duration_ns=0),
        prediction=predictions, pec=pec, repeated_learning=repeated_summary,
        sampling=dict(pec_shots_per_observable=args.pec_shots,
                      pec_total_shots_per_model=12*7*args.pec_shots,
                      validation_shots_per_observable=args.train_shots,
                      learning_repeats=args.repeats, twirl_frames_exact=4,
                      learning_cost_included='3*train_shots shared across 84 observable targets; repeated_learning includes training randomness',
                      frame_policy='fresh uniform frame history for each shot; exact frame average in generator'),
        elapsed_seconds=time.perf_counter()-start)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(prediction=predictions, pec=pec, repeated_learning=repeated_summary), indent=2), flush=True)
    print(f'Saved {out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New directory; never overwrite a run')
    parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--train-shots', type=int, default=100000)
    parser.add_argument('--pec-shots', type=int, default=20000)
    parser.add_argument('--repeats', type=int, default=100)
    run(parser.parse_args())
