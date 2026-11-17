"""ZNE (noise amplification) and PEC analysis of a SAVED SPP.

    python -m heom_spp.analyze results/some_run

Nothing is propagated here: the run directory only has to contain spp.npz and
layers.json. Expectation values are exact contractions, not samples. With
Pauli-transfer matrices O_j of the ideal layers and the Walsh form F of a
(quasi)probability over error histories,

    <O> = sum_{b_1..b_L} F(b_1..b_L) o[b_L] O_L[b_L, b_{L-1}] ... O_1[b_1, .] rho_0,

because the Pauli error P_a after layer j multiplies Pauli component b by the
character chi_b(a). F = 1 is the ideal circuit, F_p the noisy one, F_p^g the
amplified one, F_p * F_q the PEC-corrected one.

The comparison of interest is always "correlated" (the SPP itself) against
"independent" (the product of its per-layer marginals, i.e. what a layer-wise
Pauli noise model would have learned from the same device).

--baseline NAME=RUN adds further independent models taken from other runs of
the same circuit, e.g. `cli run --reset layer` or `--reset pulse` (ADOs zeroed
after every layer / every pulse block). The device is always the SPP of `run`;
a baseline only replaces the model used to amplify or to invert.
"""
import argparse
import json
from math import log, prod
from pathlib import Path
from time import perf_counter

import numpy as np

from .circuit import CircuitLayers, layer_unitary
from .mitigation import _absorb, inverse_als, real_tensor
from .mps import MPS, Truncation
from .pauli import characters, paulis, walsh
from .spp import independent_model


def pauli_transfer(unitary, basis):
    """O[b, c] = Tr(P_b U P_c U^dagger) / d (real, orthogonal)."""
    rotated = np.einsum("ji,bjk,kl->bil", unitary.conj(), basis, unitary, optimize=True)
    return np.einsum("bij,cji->bc", rotated, basis, optimize=True).real/len(unitary)


def pauli_label(word):
    if not word or set(word) - set("IXYZ"):
        raise ValueError(f"not a Pauli word: {word}")
    label = 0
    for letter in word:
        label = 4*label+"IXYZ".index(letter)
    return label


def state_vector(name, n):
    if name == "zero":
        psi = np.zeros(2**n, complex)
        psi[0] = 1
    elif name == "plus":
        psi = np.full(2**n, 2**(-n/2), complex)
    else:
        raise ValueError("initial state must be zero or plus")
    return psi


def final_paulis(factors, transfers, rho):
    """Pauli components of the final state for history weights prod(factors).

    The factors are Walsh-form MPS that multiply entrywise. Their product is
    never formed: the state carries one bond leg per factor, so memory is
    4**n * prod(bonds) instead of a core of bond prod(bonds) for every layer.
    """
    w = np.asarray(rho, float).reshape(-1, *[1]*len(factors))
    for j, transfer in enumerate(transfers):
        w = np.tensordot(transfer, w, axes=(1, 0))
        for f in factors:
            w = _absorb(w, f.cores[j])
    return w.reshape(len(rho))


def expectation(f, transfers, rho, observable):
    """Exact <O> for Walsh-form history weights f (an MPS, or a list of factors)."""
    factors = [f] if isinstance(f, MPS) else list(f)
    return float((final_paulis(factors, transfers, rho)@observable).real)


def power(f, gain, truncation):
    """f**gain entrywise, compressed. Second value: discarded norm / norm."""
    out, discarded = MPS.product([np.ones(d) for d in f.dims]), 0.
    for _ in range(gain):
        out, errors = out.hadamard(f).round(truncation)
        discarded += sum(errors)
    return out, discarded/np.sqrt(float(prod(f.dims)))


def clock(seconds):
    seconds = int(round(seconds))
    return f"{seconds//3600}:{seconds%3600//60:02d}:{seconds%60:02d}"


class Progress:
    """Progress lines: elapsed time and, where it can be estimated, time left."""

    def __init__(self, emit):
        self.emit, self.start = emit, perf_counter()

    def __call__(self, text, left=None):
        tail = "" if left is None else f" | about {clock(left)} left"
        self.emit(f"[{clock(perf_counter()-self.start)} elapsed{tail}] {text}")


def remaining(bonds, index, sweep, sweeps, per_sweep):
    """Upper estimate of the PEC time left (an inverse may converge earlier).

    per_sweep maps finished or running bonds to seconds per sweep. Bonds not
    started are extrapolated as bond**a, a from the last two bonds (2 at first).
    """
    bond = bonds[index]
    a = 2.
    if index and bonds[index-1] != bond:
        a = log(per_sweep[bond]/per_sweep[bonds[index-1]])/log(bond/bonds[index-1])
        a = min(max(a, 1.), 4.)
    later = sum(sweeps*per_sweep[bond]*(b/bond)**a for b in bonds[index+1:])
    return (sweeps-sweep)*per_sweep[bond]+later


def extrapolate(gains, values, order):
    """Polynomial through the first order+1 gains, evaluated at zero noise."""
    return float(np.polyval(np.polyfit(gains[:order+1], values[:order+1], order), 0.))


def values_at(q, histories):
    """q(d) for many histories at once."""
    out = np.ones((len(histories), 1), complex)
    for j, core in enumerate(q.cores):
        out = np.einsum("sl,slr->sr", out, core[:, histories[:, j], :].transpose(1, 0, 2))
    return out[:, 0]


def gamma_estimate(q, proposal, shots, rng):
    """Gamma = sum |q| by importance sampling from a product proposal.

    proposal[j] is a normalised distribution over the labels of layer j.
    Unbiased for any proposal with full support; the standard error is
    reported and grows if q is far from the product form.
    """
    histories = np.stack([rng.choice(len(g), size=shots, p=g) for g in proposal], axis=1)
    density = np.prod([g[histories[:, j]] for j, g in enumerate(proposal)], axis=0)
    weights = np.abs(values_at(q, histories).real)/density
    return float(weights.mean()), float(weights.std(ddof=1)/np.sqrt(shots))


def analyze(run, *, observables, initial="zero", gains=(1, 2, 3), pec_bonds=(1, 2, 4),
            zne_bond=64, sweeps=12, shots=20000, seed=20261114, progress=print,
            baselines=None, exact_entries=30_000_000):
    run = Path(run)
    loaded = MPS.load(run/"spp.npz")
    # Probabilities are real; numerical construction leaves a tiny imaginary
    # part, which is removed here and reported.
    p = real_tensor(loaded)
    imaginary = loaded.add(p.scaled(-1)).norm()
    # A compressed SPP does not sum to one exactly. Left alone, each of the L
    # marginals inherits the deficit and the layer-wise inverse compounds it
    # to total**-(L-1), which looks like a failure of independent PEC but is
    # not physics. Normalise once; the deficit is reported and judged below.
    total = float(p.total().real)
    p = p.scaled(1/total)
    plan = CircuitLayers.load(run/"layers.json")
    n = plan.n_qubits
    if p.dims != (4**n,)*len(plan.layers):
        raise ValueError("spp.npz and layers.json describe different circuits")
    basis = paulis(n)
    transfers = [pauli_transfer(layer_unitary(layer, n), basis) for layer in plan.layers]
    psi = state_vector(initial, n)
    rho = np.einsum("i,bij,j->b", psi.conj(), basis, psi).real/2**n
    vectors = {}
    for word in observables:
        if len(word) != n:
            raise ValueError(f"observable {word} must have {n} letters")
        vectors[word] = np.zeros(4**n)
        vectors[word][pauli_label(word)] = 2**n
    def measure(*factors):
        final = final_paulis(factors, transfers, rho)
        return {w: float((final@o).real) for w, o in vectors.items()}

    progress = Progress(progress)

    fp = walsh(p, n)
    independent = independent_model(p)
    # Product models: the marginals of the SPP, then the named baselines.
    models, distances = {"independent": walsh(independent, n)}, {}
    for name, path in (baselines or {}).items():
        other = real_tensor(MPS.load(Path(path)/"spp.npz"))
        if other.dims != p.dims or name in ("independent", "correlated"):
            raise ValueError(f"baseline {name} does not fit this circuit")
        other = independent_model(other)
        models[name] = walsh(other, n)
        distances[name] = other.add(independent.scaled(-1)).norm()
    fi = models["independent"]
    one = MPS.product([np.ones(4**n)]*len(p.cores))
    result = dict(run=str(run), n_qubits=n, layers=len(plan.layers), initial=initial,
        spp_bonds=list(loaded.bonds), observables=list(observables),
        discarded_imaginary_norm=imaginary, total_probability=total,
        correlation_l2=p.add(independent.scaled(-1)).norm(),
        ideal=measure(one), noisy=measure(fp), noisy_independent_model=measure(fi),
        baselines={k: str(v) for k, v in (baselines or {}).items()},
        baseline_distance_to_marginals_l2=distances)
    progress(f"loaded {run}: {n} qubits, {len(plan.layers)} layers, SPP bonds {list(p.bonds)}")
    if abs(total-1) > 1e-2*result["correlation_l2"]:
        progress(f"WARNING: the SPP sums to {total:.8f}; the missing {abs(1-total):.1e} is not small "
                 f"against the correlation signal {result['correlation_l2']:.1e}. It was normalised, "
                 "but the SPP was compressed too hard (raise --time-bond of the run or reference).")

    # ---- ZNE by probabilistic error amplification
    truncation = Truncation(zne_bond, 1e-12)
    gains = sorted(set(int(g) for g in gains))
    if not gains or gains[0] != 1:
        raise ValueError("gains must start at 1")
    zne = dict(gains=gains, correlated=[], truncation_discarded=[], **{k: [] for k in models})
    for g in gains:
        # Device noise F_p times inserted noise of gain g-1, drawn either from
        # the SPP itself or from one of the independent models. The g factors
        # are contracted side by side, which is exact; only when that state is
        # too large is the inserted noise compressed to zne_bond first.
        if 4**n*max(fp.bonds, default=1)**g <= exact_entries:
            zne["correlated"].append(measure(*[fp]*g))
            lost = 0.
        else:
            extra, lost = power(fp, g-1, truncation)
            zne["correlated"].append(measure(fp, extra))
        for name, f in models.items():
            zne[name].append(measure(fp, *[f]*(g-1)))
        zne["truncation_discarded"].append(lost)
        progress(f"ZNE gain {g} done" + (f" (inserted noise compressed, relative loss {lost:.1e})"
                                         if lost else ""))
    zne["extrapolated"] = {}
    for model in ("correlated", *models):
        zne["extrapolated"][model] = {
            f"order{order}": {w: extrapolate(gains, [e[w] for e in zne[model]], order)
                              for w in observables}
            for order in range(1, len(gains))}
    result["zne"] = zne

    # ---- PEC
    rng = np.random.default_rng(seed)
    w_inverse = characters(n)/4**n
    pec = dict(correlated=[])
    for name, f in models.items():
        marginal_f = [c[0, :, 0].real for c in f.cores]
        if min(np.min(np.abs(x)) for x in marginal_f) < 1e-8:
            raise ValueError("a layer marginal is not invertible")
        q_layers = [w_inverse@(1/x) for x in marginal_f]
        layer_gamma = [float(np.abs(q).sum()) for q in q_layers]
        corrected = fp.hadamard(MPS.product([1/x for x in marginal_f]))
        pec[name] = dict(mitigated=measure(corrected),
            gamma=float(np.prod(layer_gamma)), gamma_per_layer=layer_gamma,
            residual_rms=float(corrected.add(one.scaled(-1)).norm()
                               / np.sqrt(float(4**n)**len(p.cores))))
        if name == "independent":
            proposal = [np.abs(q)/g for q, g in zip(q_layers, layer_gamma)]
    pec_bonds = [int(b) for b in pec_bonds]
    per_sweep = {}
    for index, bond in enumerate(pec_bonds):
        began = perf_counter()

        def sweep_done(entry, index=index, bond=bond, began=began):
            per_sweep[bond] = (perf_counter()-began)/entry["sweep"]
            progress(f"PEC bond {bond}: sweep {entry['sweep']}/{sweeps}, "
                     f"residual rms {np.sqrt(entry['residual_mse']):.3e}",
                     remaining(pec_bonds, index, entry["sweep"], sweeps, per_sweep))

        q, report = inverse_als(p, n, bond=bond, sweeps=sweeps, seed=seed, progress=sweep_done)
        fq = walsh(q, n)
        gamma, error = gamma_estimate(q, proposal, shots, rng)
        last = report["history"][-1]
        pec["correlated"].append(dict(bond=bond, mitigated=measure(fp, fq),
            gamma=gamma, gamma_standard_error=error, gamma_shots=shots,
            residual_rms=float(np.sqrt(last["residual_mse"])),
            normalization_error=last["normalization_error"], sweeps=last["sweep"],
            seconds=perf_counter()-began))
        progress(f"PEC bond {bond} done: residual rms {np.sqrt(last['residual_mse']):.3e}, "
                 f"Gamma {gamma:.6f} +- {error:.1e}")
    result["seconds"] = perf_counter()-progress.start
    result["pec"] = pec
    return result


def table(result):
    words, ideal = result["observables"], result["ideal"]
    rows = [("ideal", result["ideal"]), ("noisy", result["noisy"])]
    zne = result["zne"]
    models = [k for k in result["pec"] if k != "correlated"]
    for order, values in zne["extrapolated"]["correlated"].items():
        rows.append((f"ZNE {order}, SPP", values))
        for name in models:
            rows.append((f"ZNE {order}, {name}", zne["extrapolated"][name][order]))
    for name in models:
        rows.append((f"PEC {name}", result["pec"][name]["mitigated"]))
    for entry in result["pec"]["correlated"]:
        rows.append((f"PEC SPP bond {entry['bond']}", entry["mitigated"]))
    lines = [f"{'error vs ideal':28s}" + "".join(f"{w:>12s}" for w in words) + f"{'max |err|':>12s}"]
    for name, values in rows:
        errors = [values[w]-ideal[w] for w in words]
        cells = [f"{ideal[w]:12.6f}" for w in words] if name == "ideal" else [f"{e:12.2e}" for e in errors]
        lines.append(f"{name:28s}" + "".join(cells)
                     + ("" if name == "ideal" else f"{max(abs(e) for e in errors):12.2e}"))
    pec = result["pec"]
    lines += ["", f"correlation ||p - product of marginals|| = {result['correlation_l2']:.3e}"]
    for name, distance in result["baseline_distance_to_marginals_l2"].items():
        lines.append(f"||{name} - product of marginals|| = {distance:.3e}")
    for name in models:
        lines.append(f"Gamma, {name} PEC = {pec[name]['gamma']:.6f}   "
                     f"residual rms {pec[name]['residual_rms']:.2e}")
    for entry in pec["correlated"]:
        lines.append(f"Gamma, SPP PEC bond {entry['bond']} = {entry['gamma']:.6f} "
                     f"+- {entry['gamma_standard_error']:.1e}   residual rms {entry['residual_rms']:.2e}")
    return "\n".join(lines)


def default_observables(n):
    single = ["I"*q+"Z"+"I"*(n-q-1) for q in range(n)]
    pairs = ["I"*q+"ZZ"+"I"*(n-q-2) for q in range(n-1)]
    return single+pairs+["X"+"I"*(n-1)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="directory with spp.npz and layers.json")
    parser.add_argument("--output", type=Path, help="default: <run>/mitigation.json")
    parser.add_argument("--observables", help="comma-separated Pauli words, q0 first, e.g. ZIII,ZZII")
    parser.add_argument("--initial", choices=("zero", "plus"), default="zero")
    parser.add_argument("--gains", default="1,2,3", help="integer noise gains for ZNE")
    parser.add_argument("--pec-bonds", default="1,2,4", help="time bond of the correction q")
    parser.add_argument("--zne-bond", type=int, default=64,
                        help="only used when the exact ZNE contraction would not fit in memory")
    parser.add_argument("--sweeps", type=int, default=12)
    parser.add_argument("--shots", type=int, default=20000, help="samples for the Gamma estimate")
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--baseline", action="append", default=[], metavar="NAME=RUN",
                        help="extra independent model from another run, e.g. reset-layer=results/x")
    args = parser.parse_args(argv)
    n = CircuitLayers.load(args.run/"layers.json").n_qubits
    observables = args.observables.split(",") if args.observables else default_observables(n)
    result = analyze(args.run, observables=observables, initial=args.initial,
        gains=[int(g) for g in args.gains.split(",")],
        pec_bonds=[int(b) for b in args.pec_bonds.split(",") if b],
        zne_bond=args.zne_bond, sweeps=args.sweeps, shots=args.shots, seed=args.seed,
        baselines=dict(item.split("=", 1) for item in args.baseline))
    output = args.output or args.run/"mitigation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print()
    print(table(result))
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
