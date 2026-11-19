"""Bounded two-slot I/Z process benchmark, NOT a general SPP/HEOM interface.

Four-qubit ideal Ising controls alternate with two equal dephasing slots.
Only qubit 0 couples to one oscillator. Controls, Pauli frames, preparation,
and readout are instantaneous and ideal. The oscillator persists across slots.
No fitted function receives latent error histories or the generating bath.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, linprog
from .core import I, X, Y, Z, embed
from .quantum import bosonic_bath, mode_parameters

# History order 00, 01, 10, 11 (first slot is the high bit).
# Terminal calibration: first-slot, second-slot, and parity X/Z responses.
CALIBRATION = np.array([[1, 1, -1, -1], [1, -1, 1, -1],
                        [1, -1, -1, 1]], dtype=float)


@dataclass(frozen=True)
class ProcessConfig:
    duration: float = 12.0  # ns per noise slot
    sigma: float = 0.06  # rad/ns, illustrative, not device calibrated
    beta: float = 60.0  # ns
    omega_min: float = 0.03
    omega_max: float = 0.12  # one log midpoint omega=0.06 rad/ns

    def bath(self):
        return dict(modes=1, **{k: v for k, v in asdict(self).items()
                               if k != 'duration'})


def heom_solver(config, depth=8, n=1, tolerance=1e-10):
    import qutip as qt
    from qutip.solver.heom import HEOMSolver
    if n not in (1, 4) or not 1 <= depth <= 12:
        raise ValueError('Bounded benchmark permits n=1/4 and depth=1..12')
    bath, _, _, _ = bosonic_bath(embed(Z, (0,), n)/2, **config.bath())
    return HEOMSolver(qt.Qobj(np.zeros((2**n, 2**n))), bath, depth,
                      options=dict(store_ados=True, store_states=True,
                                   normalize_output=False, progress_bar='',
                                   atol=tolerance, rtol=10*tolerance,
                                   nsteps=20000, method='adams'))


def evolve_ados(solver, physical_ados, duration):
    """Apply the same physical intervention to ALL ADOs before this call.

QuTiP 5.x array restart stores each ADO transposed (column vectorization).
Use public extract() on return; a split-evolution regression protects layout.
Unnormalized branches must never be normalized by the solver.
"""
    state = np.ascontiguousarray(physical_ados.swapaxes(-1, -2))
    result = solver.run(state, [0., duration]).ado_states[-1]
    return np.array([result.extract(i).full()
                     for i in range(len(solver.ados.labels))])


def initial_ados(solver, rho):
    result = np.zeros((len(solver.ados.labels), *rho.shape), complex)
    result[0] = rho
    return result


def heom_probabilities(config=ProcessConfig(), depth=8, tolerance=1e-10):
    """Diagonal process coefficients via X measurement/repreparation testers.

For U_SB=I tensor K0 + Z tensor K1, preparing |+>, observing X=(-1)^a,
then resetting to |+> implements K_a on the bath. This is a simulation-only
coefficient extractor, distinct from the terminal-count learning protocol.
"""
    solver = heom_solver(config, depth, tolerance=tolerance)
    plus = np.ones(2)/np.sqrt(2)
    first = evolve_ados(solver, initial_ados(solver, np.outer(plus, plus)),
                        config.duration)
    probabilities = []
    for a in (0, 1):
        eigen = np.array([1., (-1.)**a])/np.sqrt(2)
        reset = np.outer(plus, eigen)
        branch = reset @ first @ reset.conj().T
        second = evolve_ados(solver, branch, config.duration)
        for b in (0, 1):
            e = np.array([1., (-1.)**b])/np.sqrt(2)
            probabilities.append(float(np.vdot(e, second[0] @ e).real))
    return np.array(probabilities), dict(depth=depth, ado_count=len(first))


def explicit_probabilities(config=ProcessConfig(), fock=10):
    """Independent bath-only block expansion, no HEOM or Pauli fit."""
    if not 2 <= fock <= 32:
        raise ValueError('Fock cutoff must be 2..32')
    w, g, nb = mode_parameters(**config.bath())
    a = np.diag(np.sqrt(np.arange(1, fock)), 1)
    hb = w[0] * (a.conj().T @ a)
    v = g[0] * (a+a.conj().T)/2
    up, um = (expm(-1j*config.duration*(hb+s*v)) for s in (1, -1))
    k = [(up+um)/2, (up-um)/2]
    weights = (nb[0]/(nb[0]+1))**np.arange(fock)
    rho = np.diag(weights/weights.sum())
    p = np.array([np.trace(kb @ ka @ rho @ ka.conj().T @ kb.conj().T).real
                  for ka in k for kb in k])
    return p, dict(fock=fock, initial_thermal_tail=float((nb[0]/(nb[0]+1))**fock))


def checked_probability(p, tolerance=2e-8):
    """Reject solver failures; only repair roundoff, with raw values archived."""
    p = np.asarray(p, float)
    if p.shape != (4,) or not np.all(np.isfinite(p)) or p.min() < -tolerance or abs(p.sum()-1) > tolerance:
        raise ValueError(f'Invalid process probabilities: {p}')
    p = np.maximum(p, 0)
    return p/p.sum()


def independent_projection(p):
    table = np.asarray(p).reshape(2, 2)
    return np.outer(table.sum(axis=1), table.sum(axis=0)).ravel()


def fit_terminal_counts(minus_counts, shots, independent=False):
    """Binomial maximum likelihood on terminal counts only, no truth argument.

Independent model is fitted to exactly the same three datasets as the joint
model (2 versus 3 parameters), not privileged single-slot calibrations.
"""
    k = np.asarray(minus_counts, float)
    if k.shape != (3,) or shots < 1 or np.any(k < 0) or np.any(k > shots):
        raise ValueError('Three valid terminal minus counts required')

    def distribution(x):
        if independent:
            r, s = x
            return np.array([(1-r)*(1-s), (1-r)*s, r*(1-s), r*s])
        return x

    def loss(x):
        minus = np.clip((1-CALIBRATION @ distribution(x))/2, 1e-14, 1-1e-14)
        return -float(np.sum(k*np.log(minus)+(shots-k)*np.log1p(-minus)))/shots

    if independent:
        x0 = np.clip(k[:2]/shots, 1e-5, 1-1e-5)
        result = minimize(loss, x0, method='L-BFGS-B', bounds=[(0, 1)]*2,
                          options=dict(ftol=1e-14, gtol=1e-8, maxiter=1000))
    else:
        # The three nontrivial Walsh coefficients plus normalization have rank 4.
        candidate = np.linalg.solve(np.vstack([np.ones(4), CALIBRATION]),
                                    np.r_[1, 1-2*k/shots])
        if candidate.min() >= 0:
            return candidate, dict(success=True, nll=loss(candidate)*shots,
                                   boundary=False)
        result = minimize(loss, np.full(4, .25), method='SLSQP', bounds=[(0, 1)]*4,
                          constraints=[dict(type='eq', fun=lambda x: x.sum()-1)],
                          options=dict(ftol=1e-12, maxiter=1000))
    if not result.success:
        raise RuntimeError(f'Terminal fit failed: {result.message}')
    p = checked_probability(distribution(result.x))
    return p, dict(success=True, nll=loss(result.x)*shots, boundary=bool(p.min() < 1e-8))


def convolution_matrix(p):
    """Correction b changes history a to a XOR b at the SAME noise slots."""
    indices = np.arange(4)
    return np.asarray(p)[indices[:, None] ^ indices[None, :]]


def inverse_qp(p):
    """min ||q||_1 subject to C(p)q=delta_00, a signed history convolution.

This is not a CPTP inverse, an MPO algorithm, or a general quantum comb inverse.
Corrections are ideal I/Z0 after each noise slot. They do not change the bath.
"""
    matrix = convolution_matrix(p)
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > 1e8:
        raise ValueError('SPP inverse is singular/ill-conditioned')
    target = np.array([1., 0, 0, 0])
    result = linprog(np.ones(8), A_eq=np.c_[matrix, -matrix], b_eq=target,
                     bounds=(0, None), method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    q = result.x[:4]-result.x[4:]
    residual = float(np.max(np.abs(matrix @ q-target)))
    if residual > 1e-8:
        raise RuntimeError('Inverse residual too large')
    gamma = float(np.abs(q).sum())
    return q, dict(gamma=gamma, gamma_squared=gamma**2, condition=condition,
                   residual_inf=residual,
                   min_abs_fourier=float(np.min(np.abs(np.linalg.eigvalsh(matrix)))))


def ising_circuits(seed=20261029, count=12):
    """Held out from calibration: U_l=exp[-i t_l(J_l sum ZZ+h_l sum X)].

The Ising time t is dimensionless simulated time, NOT physical gate duration.
Every control has zero physical duration in this bounded process benchmark.
"""
    rng = np.random.default_rng(seed)
    zz = sum(embed(np.kron(Z, Z), (j, j+1), 4) for j in range(3))
    xx = sum(embed(X, (j,), 4) for j in range(4))
    circuits, parameters = [], []
    for _ in range(count):
        params = np.c_[rng.uniform(.25, 1.2, 2), rng.uniform(.6, 1.1, 2),
                       rng.uniform(.25, 1.05, 2)]
        circuits.append([expm(-1j*t*(j*zz+h*xx)) for t, j, h in params])
        parameters.append(params.tolist())
    observables = [embed(X, (j,), 4) for j in range(4)]
    observables += [embed(np.kron(Z, Z), (j, j+1), 4) for j in range(3)]
    return circuits, np.asarray(observables), parameters


def history_states(controls):
    z0 = embed(Z, (0,), 4)
    states = []
    for a in range(4):
        psi = np.ones(16, complex)/4
        for layer, u in enumerate(controls):
            psi = u @ psi
            if (a >> (1-layer)) & 1:
                psi = z0 @ psi
        states.append(np.outer(psi, psi.conj()))
    return np.asarray(states)


def response_matrix(circuits, observables):
    return np.array([np.einsum('oij,hji->oh', observables, history_states(c)).real
                     for c in circuits])


def twirled_heom_state(controls, config=ProcessConfig(), depth=8,
                       correction=0, twirl=True):
    """Direct 4-qubit HEOM, independent of the extracted probability table.

Average X0-conjugations around EACH dephasing slot (4 frame histories).
For pure Z coupling, I/X is the exact quotient of the full Pauli twirl:
Z conjugations commute with the slot propagator and add no new frames.
"""
    solver = heom_solver(config, depth, n=4)
    x0, z0 = embed(X, (0,), 4), embed(Z, (0,), 4)
    total = np.zeros((16, 16), complex)
    frames = range(4) if twirl else [0]
    for frame in frames:
        ados = initial_ados(solver, np.ones((16, 16), complex)/16)
        for layer, u in enumerate(controls):
            f = x0 if (frame >> (1-layer)) & 1 else np.eye(16)
            op = f @ u
            ados = op @ ados @ op.conj().T
            ados = evolve_ados(solver, ados, config.duration)
            op = (z0 if (correction >> (1-layer)) & 1 else np.eye(16)) @ f
            ados = op @ ados @ op.conj().T
        total += ados[0]/len(frames)
    return total


def pec_statistics(response, truth, q, shots, rng):
    """Sample correction AND +/-1 measurement outcomes; save raw counts.

Y=gamma sign(q_b) m. E[Y^2]=gamma^2 for a Pauli measurement. Variance
below is conditional on fitted q; repeated learning is handled by the runner.
"""
    gamma = np.abs(q).sum()
    correction_means = response @ convolution_matrix(truth)
    mu = correction_means @ q
    joint = (np.abs(q)/gamma)[:, None] * np.stack(
        [(1+correction_means)/2, (1-correction_means)/2], axis=-1)
    counts = rng.multinomial(shots, np.clip(joint.ravel(), 0, 1)/joint.sum()).reshape(4, 2)
    estimate = gamma * np.sum(np.sign(q)*(counts[:, 0]-counts[:, 1]))/shots
    return float(estimate), float(mu), float((gamma**2-mu**2)/shots), counts
