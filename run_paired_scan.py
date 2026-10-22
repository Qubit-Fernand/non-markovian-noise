#!/usr/bin/env python3
"""Synthetic paired-pulse scan; does not fit or claim real-device noise.

Example: python run_paired_scan.py --output results/paired_scan_20261015
Outputs include raw trajectory endpoints, calibration points, independent-seed
confirmations, time-step diagnostics, code snapshots, and PNG/PDF figures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import scipy
from nmnoise.paired import (paired_schedules, pulse_schedule, shared_grid,
                           integrated_ou_noise, propagate_bloch, instantaneous_coherence,
                           fit_lindblad_rate, summarize_pair)


def evaluate(total, width, taus, sigma, samples, dt, seed):
    schedules = paired_schedules(total, width)
    grid = shared_grid(schedules, dt)
    noise = integrated_ou_noise(grid, taus, samples, sigma, seed)
    bloch = propagate_bloch(schedules, grid, noise)
    result = summarize_pair(bloch)
    result.update(total_ns=total, width_ns=width, taus_ns=list(taus), sigma_rad_per_ns=sigma,
                  samples=samples, pulse_dt_ns=dt, seed=seed)
    result['instantaneous_x'] = [instantaneous_coherence(total, total*np.array(c), taus, sigma)
                                  for c in ((.25, .75), (.125, .625))]
    result['instantaneous_delta_x'] = result['instantaneous_x'][0]-result['instantaneous_x'][1]
    return result, bloch, grid, noise


def calibrate(taus, sigma, width, samples, dt, seed):
    """Six independent training circuits: Ramsey + finite echo at 3 durations."""
    points, raw, times, xs = [], {}, [], []
    for j, total in enumerate((100., 400., 1600.)):
        schedules = [pulse_schedule(total, width, []), pulse_schedule(total, width, [total/2])]
        grid = shared_grid(schedules, dt)
        noise = integrated_ou_noise(grid, taus, samples, sigma, seed+j)
        bloch = propagate_bloch(schedules, grid, noise)
        raw[f't{int(total)}'] = bloch
        for k, name in enumerate(('Ramsey', 'echo')):
            x = bloch[k, :, 0]
            points.append(dict(sequence=name, total_ns=total, x=float(x.mean()),
                               mc_se=float(x.std(ddof=1)/np.sqrt(samples)), seed=seed+j))
            times.append(total)
            xs.append(float(x.mean()))
    fitted = fit_lindblad_rate(times, xs)
    fitted.update(points=points, samples=samples,
                  method='One shared rate; equal-sequence expected binomial NLL on synthetic X probabilities.')
    return fitted, raw


def coupled_grid_check(total, width, taus, sigma, samples, coarse_dt, seed):
    """Both resolutions use the same exact OU integrals, aggregated to coarse."""
    schedules = paired_schedules(total, width)
    fine = shared_grid(schedules, coarse_dt/2)
    coarse = shared_grid(schedules, coarse_dt)
    noise = integrated_ou_noise(fine, taus, samples, sigma, seed)
    indices = np.searchsorted(fine, coarse)
    np.testing.assert_allclose(fine[indices], coarse, atol=1e-9)
    cumulative = np.c_[np.zeros(samples), np.cumsum(noise*np.diff(fine), axis=1)]
    rebinned = np.diff(cumulative[:, indices], axis=1)/np.diff(coarse)
    fine_bloch = propagate_bloch(schedules, fine, noise)
    coarse_bloch = propagate_bloch(schedules, coarse, rebinned)
    change = ((fine_bloch[0, :, 0]-fine_bloch[1, :, 0])
              -(coarse_bloch[0, :, 0]-coarse_bloch[1, :, 0]))
    return dict(coarse_dt_ns=coarse_dt, fine_dt_ns=coarse_dt/2, seed=seed, samples=samples,
                delta_change=float(change.mean()),
                delta_change_mc_se=float(change.std(ddof=1)/np.sqrt(samples)),
                max_bloch_mean_change=float(np.abs((fine_bloch-coarse_bloch).mean(axis=1)).max())), fine_bloch


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def write_manifest(out):
    manifest = []
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name != 'SHA256SUMS':
            manifest.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(out)}')
    (out/'SHA256SUMS').write_text('\n'.join(manifest)+'\n')


def make_plots(out, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 16, 'text.usetex': True})
    rows = report['scan']
    sigmas, totals, taus = (report['config'][key] for key in ('sigmas', 'totals_ns', 'single_taus_ns'))
    columns = 2
    panels = int(np.ceil(len(sigmas)/columns))
    fig, axes = plt.subplots(panels, columns, figsize=(13, 4.9*panels), squeeze=False, layout='constrained')
    limit = max(.01, max(abs(row['delta_x']) for row in rows))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    for ax, sigma in zip(axes.flat, sigmas):
        matrix = np.array([[next(r['delta_x'] for r in rows if r['sigma_rad_per_ns']==sigma
                                 and r['total_ns']==total and r['profile']==f'ou_{tau:g}')
                            for total in totals] for tau in taus])
        im = ax.imshow(matrix, origin='lower', aspect='auto', cmap='RdBu_r', norm=norm)
        ax.set_xticks(range(len(totals)), [f'{t:g}' for t in totals])
        ax.set_yticks(range(len(taus)), [f'{t:g}' for t in taus])
        ax.set_xlabel(r'Total duration $T$ (ns)')
        ax.set_ylabel(r'OU correlation time $\tau_c$ (ns)')
        ax.set_title(rf'$\sigma={sigma:g}$ rad/ns')
        for i in range(len(taus)):
            for j in range(len(totals)):
                source = next(r for r in rows if r['sigma_rad_per_ns']==sigma
                              and r['total_ns']==totals[j] and r['profile']==f'ou_{taus[i]:g}')
                uncertain = abs(source['delta_x']) < 3*source['delta_x_mc_se']
                label = f'{matrix[i,j]:.3f}'+('*' if uncertain else '')
                ax.text(j, i, label, ha='center', va='center', fontsize=12,
                        color='white' if abs(matrix[i,j])>.65*limit else 'black')
    for ax in list(axes.flat)[len(sigmas):]:
        ax.set_visible(False)
    fig.colorbar(im, ax=list(axes.flat), label=r'Paired contrast $\Delta X=X_A-X_B$', shrink=.82)
    fig.suptitle(rf'Synthetic OU scan: finite $X_\pi$ pulses ({report["config"]["width_ns"]:g} ns)'
                 '\n'+r'Lindblad: $\Delta X=0$; * means $|\Delta X|<3$ Monte Carlo SE')
    fig.savefig(out/'pair_contrast.png', dpi=180)
    fig.savefig(out/'pair_contrast.pdf')
    plt.close(fig)
    confirmed = report['confirmations']
    if not confirmed:
        return
    fig, axes = plt.subplots(1, 2, figsize=(19, 6.2), layout='constrained')
    xx = np.arange(len(confirmed))
    labels = [rf'$T={r["total_ns"]:g}$'+'\n'+
              (rf'$\tau_c={r["taus_ns"][0]:g}$' if len(r['taus_ns'])==1 else 'Mix12')+'\n'+
              rf'$\sigma={r["sigma_rad_per_ns"]:g}$' for r in confirmed]
    for j, (name, color, offset) in enumerate((('A', '#246a92', -.16), ('B', '#bd6934', .16))):
        values = [r['bloch_mean'][j][0] for r in confirmed]
        errors = [r['bloch_mc_se'][j][0] for r in confirmed]
        axes[0].errorbar(xx+offset, values, yerr=errors, fmt='o', color=color, capsize=4,
                         label=f'Colored {name} (1 MC SE)')
    axes[0].plot(xx, [r['lindblad_x'] for r in confirmed], 'k_', markersize=24, label='Fitted Lindblad A=B')
    axes[0].set_xticks(xx, labels, fontsize=13)
    axes[0].set_ylabel(r'Final $\langle X\rangle$')
    axes[0].set_ylim(-.05, 1.05)
    axes[0].legend(fontsize=12, loc='best')
    axes[0].set_title('Independent-seed confirmation')
    axes[1].errorbar(xx, [r['delta_x'] for r in confirmed],
                     yerr=[r['delta_x_mc_se'] for r in confirmed], fmt='o', color='#246a92',
                     capsize=4, label='Finite pulses (1 paired MC SE)')
    axes[1].plot(xx, [r['instantaneous_delta_x'] for r in confirmed], 'x', color='#bd6934',
                 markersize=9, label='Instantaneous-pulse analytic reference')
    axes[1].axhline(0, color='black', ls='--', label='Lindblad')
    axes[1].set_xticks(xx, labels, fontsize=13)
    axes[1].set_ylabel(r'$\Delta X=X_A-X_B$')
    axes[1].set_title('Timing contrast, not model-fit error')
    axes[1].legend(fontsize=12, loc='best')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_xlabel(r'$T,\tau_c$ in ns; $\sigma$ in rad/ns', fontsize=13)
    fig.savefig(out/'confirmed_pairs.png', dpi=180)
    fig.savefig(out/'confirmed_pairs.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=2048)
    parser.add_argument('--calibration-samples', type=int, default=8192)
    parser.add_argument('--confirmation-samples', type=int, default=16384)
    parser.add_argument('--pulse-dt', type=float, default=1.)
    parser.add_argument('--width', type=float, default=20.)
    parser.add_argument('--sigmas', type=float, nargs='+', default=[.001, .002, .004, .008])
    parser.add_argument('--totals', type=float, nargs='+', default=[200., 400., 800., 1600., 3200.])
    parser.add_argument('--taus', type=float, nargs='+', default=[10., 30., 100., 300., 1000., 3000., 10000.])
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--skip-confirmation', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    out = args.output
    if args.plot_only:
        report = json.loads((out/'summary.json').read_text())
        make_plots(out, report)
        # Keep the original numerical source; record subsequent figure changes.
        content = Path(__file__).read_bytes()
        (out/'source'/'plotting_revision.py').write_bytes(content)
        report['plot_revision_sha256'] = hashlib.sha256(content).hexdigest()
        save_json(out/'summary.json', report)
        write_manifest(out)
        return
    if out.exists():
        parser.error('Use a new output directory; existing data will not be overwritten')
    if min(args.samples, args.calibration_samples, args.confirmation_samples) < 2:
        parser.error('At least two trajectories are required')
    # Validate all schedules before creating output.
    for total in args.totals:
        paired_schedules(total, args.width)
    out.mkdir(parents=True)
    profiles = [(f'ou_{tau:g}', [tau]) for tau in args.taus]
    profiles.append(('mixture12', np.geomspace(10., 10000., 12).tolist()))
    report = dict(config=dict(samples=args.samples, calibration_samples=args.calibration_samples,
                  confirmation_samples=args.confirmation_samples, pulse_dt_ns=args.pulse_dt,
                  width_ns=args.width, sigmas=args.sigmas, totals_ns=args.totals,
                  single_taus_ns=args.taus, seed=args.seed),
                  assumptions=['Synthetic zero-mean stationary independent OU modes, known noise parameters.',
                  'One qubit initially |+>; ideal state preparation/readout; rectangular X-pi control.',
                  'No T1, leakage, amplitude error, quantum bath or hardware data.',
                  'Lindblad shared rate fit to Ramsey/echo at 100,400,1600 ns, never to paired tests.',
                  'MC SE uses shared noise for A/B; it is not a hardware shot error or parameter-confidence interval.',
                  'OU interval integrals are sampled exactly; only driven-cell time ordering is approximated.'],
                  calibrations={}, scan=[], confirmations=[], pulse_width_checks=[], controls={},
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  source_sha256={})
    source_dir = out/'source'
    source_dir.mkdir()
    for name in ('run_paired_scan.py', 'nmnoise/paired.py', 'nmnoise/core.py'):
        content = Path(name).read_bytes()
        dest = source_dir/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        report['source_sha256'][name] = hashlib.sha256(content).hexdigest()
    report['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    raw, calibration_raw, confirmation_raw = {}, {}, {}
    start = time.perf_counter()
    for p, (profile, taus) in enumerate(profiles):
        for s, sigma in enumerate(args.sigmas):
            key = f'{profile}_sigma{sigma:g}'
            fit, training = calibrate(taus, sigma, args.width, args.calibration_samples,
                                      args.pulse_dt, args.seed+100000+p*1000+s*10)
            report['calibrations'][key] = fit
            calibration_raw.update({f'{key}_{k}': v for k,v in training.items()})
            for t, total in enumerate(args.totals):
                seed = args.seed+10000+p*1000+s*100+t
                row, bloch, _, _ = evaluate(total, args.width, taus, sigma, args.samples, args.pulse_dt, seed)
                row.update(profile=profile, calibration_key=key, id=f'{key}_T{total:g}',
                           lindblad_x=float(np.exp(-fit['gamma_phi_per_ns']*total)))
                row['lindblad_delta_x'] = 0.
                row['max_model_x_gap'] = float(np.max(np.abs(bloch.mean(axis=1)[:,0]-row['lindblad_x'])))
                report['scan'].append(row)
                raw[row['id']] = bloch
            print(key, 'max |delta X|=', round(max(abs(r['delta_x']) for r in report['scan'] if r['calibration_key']==key),4), flush=True)
    np.savez_compressed(out/'scan_bloch.npz', **raw)
    np.savez_compressed(out/'calibration_bloch.npz', **calibration_raw)
    # Penalize noisy exploratory maxima, then re-evaluate with independent seeds.
    # This score is a screening heuristic, NOT a simultaneous confidence bound.
    def selection_score(row):
        return abs(row['delta_x'])-3*row['delta_x_mc_se']

    selected = [max((r for r in report['scan'] if r['sigma_rad_per_ns']==sigma and r['profile']!='mixture12'),
                    key=selection_score) for sigma in args.sigmas]
    selected.append(max((r for r in report['scan'] if r['profile']=='mixture12'), key=selection_score))
    # Predefined follow-ups for the previous report's noise strength/spectrum:
    # short, intermediate, and long experiments; retain this order for seed stability.
    for total in (200., 1600., 3200.):
        for row in report['scan']:
            if row['profile']=='mixture12' and row['sigma_rad_per_ns']==.004 and row['total_ns']==total:
                if row['id'] not in [r['id'] for r in selected]:
                    selected.append(row)
    report['selection_rule'] = ('Best abs(delta)-3*MC_SE per sigma for single OU, best mixture overall, '
                               'plus mixture sigma=.004 at T=200,1600,3200 if present; exploratory, '
                               'not a multiple-comparison significance test.')
    if not args.skip_confirmation:
        for j, candidate in enumerate(selected):
            total, taus, sigma = candidate['total_ns'], candidate['taus_ns'], candidate['sigma_rad_per_ns']
            row, bloch, grid, noise = evaluate(total, args.width, taus, sigma, args.confirmation_samples,
                                               args.pulse_dt/2, args.seed+200000+j*100)
            row.update(profile=candidate['profile'], candidate_id=candidate['id'], lindblad_x=candidate['lindblad_x'])
            row['grid_check'], fine_bloch = coupled_grid_check(total, args.width, taus, sigma,
                  args.confirmation_samples, args.pulse_dt/2, args.seed+300000+j*100)
            static_seed = args.seed+600000+j*100
            fixed = np.random.default_rng(static_seed).normal(scale=sigma, size=(args.confirmation_samples,1))
            static = propagate_bloch(paired_schedules(total,args.width), grid,
                                      np.broadcast_to(fixed,(len(fixed),len(grid)-1)))
            row['quasistatic_control'] = dict(summarize_pair(static), seed=static_seed)
            report['confirmations'].append(row)
            confirmation_raw[f'case{j}_bloch'] = bloch
            confirmation_raw[f'case{j}_quasistatic_bloch'] = static
            confirmation_raw[f'case{j}_grid_check_fine_bloch'] = fine_bloch
            # Retain 32 actual example trajectories. All final Bloch samples are
            # saved; full input draws are reproducible from seed + source snapshot.
            np.savez_compressed(out/f'confirmation_noise_examples_{j}.npz', times=grid,
                                interval_average_noise=noise[:32], total_trajectories=args.confirmation_samples)
            for w, width in enumerate((10., 20., 40.)):
                width_row, width_bloch, _, _ = evaluate(total, width, taus, sigma, args.confirmation_samples,
                          args.pulse_dt/2, args.seed+400000+j*100+w)
                width_row['candidate_id'] = candidate['id']
                report['pulse_width_checks'].append(width_row)
                confirmation_raw[f'case{j}_width{width:g}'] = width_bloch
            print('confirmed', candidate['id'], 'delta=', row['delta_x'], '+/-', row['delta_x_mc_se'], flush=True)
        # Finite-pulse quasi-static and zero-noise controls, same durations/width.
        candidate = max(selected, key=lambda r: abs(r['delta_x']))
        schedules = paired_schedules(candidate['total_ns'], args.width)
        grid = shared_grid(schedules, args.pulse_dt/2)
        fixed = np.random.default_rng(args.seed+500000).normal(scale=candidate['sigma_rad_per_ns'],
                                                             size=(args.confirmation_samples, 1))
        static = propagate_bloch(schedules, grid, np.broadcast_to(fixed, (len(fixed),len(grid)-1)))
        report['controls']['quasistatic'] = dict(summarize_pair(static), candidate_id=candidate['id'],
                                                seed=args.seed+500000)
        zero = propagate_bloch(schedules, grid, np.zeros((2,len(grid)-1)))
        report['controls']['zero_noise'] = summarize_pair(zero)
        confirmation_raw['quasistatic'] = static
        np.savez_compressed(out/'confirmation_bloch.npz', **confirmation_raw)
    report['elapsed_seconds_before_plotting'] = time.perf_counter()-start
    save_json(out/'summary.json', report)
    make_plots(out, report)
    # Portable per-run integrity manifest; include raw data, plots and source.
    write_manifest(out)
    print('Saved', out.resolve(), flush=True)


if __name__ == '__main__':
    main()
