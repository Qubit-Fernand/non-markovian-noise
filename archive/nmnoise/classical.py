"""Continuous colored-noise trajectories and a simultaneous Lindblad baseline.

C(t) = sum_k v_k exp(-gamma_k |t|)
S(omega) = sum_k 2 v_k gamma_k / (gamma_k**2 + omega**2).
Equal v_k on logarithmically spaced gamma_k approximate 1/|omega| only inside
the rate band; the spectrum is regularized (finite variance) outside it.
"""
import numpy as np
from scipy.linalg import expm
from .core import Z, embed, ket0, timeline


def ou_noise(pulses, n, dt, samples=256, sigma=0.004, modes=12,
             gamma_min=1e-4, gamma_max=0.1, seed=20260921, reset=False,
             static=False):
    if samples < 1 or modes < 1 or sigma < 0 or not (0 < gamma_min <= gamma_max):
        raise ValueError("Invalid noise parameters")
    ids = timeline(pulses, dt)
    rng = np.random.default_rng(seed)
    if static:
        delta = rng.normal(scale=sigma, size=(samples, n))
        return np.broadcast_to(delta[:, None, :], (samples, len(ids), n)).copy()
    rates = np.geomspace(gamma_min, gamma_max, modes)
    std = sigma / np.sqrt(modes)
    decay = np.exp(-rates * dt)
    innovation = std * np.sqrt(-np.expm1(-2*rates*dt))
    # Stationary initial condition, not an artificial zero-noise startup.
    modes_t = rng.normal(scale=std, size=(samples, n, modes))
    values = np.empty((samples, len(ids), n))
    for t in range(len(ids)):
        if t and reset and ids[t] != ids[t-1]:
            modes_t = rng.normal(scale=std, size=modes_t.shape)
        values[:, t, :] = modes_t.sum(axis=-1)
        modes_t = decay * modes_t + innovation * rng.normal(size=modes_t.shape)
    return values


def dense_trajectories(pulses, noise, dt, initial=None):
    """Exact exponential for H_control + H_noise on each sampled time cell.

The cellwise noise approximation still needs dt convergence. Bath history is
never reset by this propagator; it is encoded in the provided full trajectories.
"""
    samples, steps, n = noise.shape
    ids = timeline(pulses, dt)
    if steps != len(ids):
        raise ValueError("Noise length does not match schedule")
    psi0 = ket0(n) if initial is None else np.asarray(initial, complex)
    psi = np.broadcast_to(psi0, (samples, 2**n)).copy()
    controls = [embed(p.h, p.sites, n) for p in pulses]
    z_diags = np.array([np.diag(embed(Z, (q,), n)).real for q in range(n)])
    ii = np.arange(2**n)
    for t, k in enumerate(ids):
        h = np.broadcast_to(controls[k], (samples, 2**n, 2**n)).copy()
        h[:, ii, ii] += 0.5 * noise[:, t, :] @ z_diags
        eig, vec = np.linalg.eigh(h)
        coeff = np.einsum("bij,bi->bj", vec.conj(), psi)
        psi = np.einsum("bij,bj->bi", vec, np.exp(-1j*dt*eig)*coeff)
    return psi


def lindblad(pulses, n, gamma_phi, initial=None):
    """d rho/dt = -i[H,rho] + gamma_phi/2 sum(Z rho Z-rho).

One-qubit free coherence decays at gamma_phi. Controls and dissipation act
simultaneously; no gate-end noise insertion is used.
"""
    if gamma_phi < 0:
        raise ValueError("Negative dephasing rate")
    d = 2**n
    psi = ket0(n) if initial is None else np.asarray(initial, complex)
    rho = np.outer(psi, psi.conj())
    identity = np.eye(d)
    dissipator = np.zeros((d*d, d*d), complex)
    for q in range(n):
        z = embed(Z, (q,), n)
        dissipator += gamma_phi/2 * (np.kron(z.conj(), z)-np.eye(d*d))
    for p in pulses:
        h = embed(p.h, p.sites, n)
        liouvillian = -1j*(np.kron(identity, h)-np.kron(h.T, identity))+dissipator
        rho = (expm(p.duration*liouvillian) @ rho.ravel(order="F")).reshape((d,d), order="F")
    return rho


def ou_chi(t, sigma=0.004, modes=12, gamma_min=1e-4, gamma_max=0.1):
    """Exact one-qubit Ramsey exponent for the continuous OU mixture."""
    rates = np.geomspace(gamma_min, gamma_max, modes)
    return float(sigma**2/modes * np.sum((rates*t+np.expm1(-rates*t))/rates**2))
