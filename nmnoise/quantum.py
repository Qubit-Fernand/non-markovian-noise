"""Genuine quantum Gaussian bath: HEOM versus explicit thermal oscillators.

The small benchmark discretizes a band-limited 1/omega SYMMETRIZED bath
spectrum using log-midpoint quadrature. It is not the paper's continuum BSD
decomposition. With only two modes it is a solver validation toy, not a
converged continuum 1/f prediction. Increasing modes, HEOM depth and bandwidth
resolution are independent convergence questions.
"""
from math import comb
import numpy as np
from .core import Pulse, X, Y, Z


def quantum_schedule():
    return [Pulse("Ry(pi/2)", 10., (0,), np.pi/40*Y),
            Pulse("idle", 10., (0,), np.zeros((2,2), complex)),
            Pulse("Rx(pi/2)", 10., (0,), np.pi/40*X),
            Pulse("idle", 10., (0,), np.zeros((2,2), complex))]


def mode_parameters(modes=2, omega_min=0.015, omega_max=0.24,
                    sigma=0.03, beta=60.):
    """beta=hbar/(k_B T) in ns; illustrative parameters, not a device fit.

Integral of the symmetrized spectrum / (2*pi) is sigma**2. Uniform variance
per log bin is a quadrature of 1/omega on the specified finite band.
"""
    if modes < 1 or not (0 < omega_min < omega_max) or beta <= 0 or sigma < 0:
        raise ValueError("Invalid quantum bath parameters")
    edges = np.geomspace(omega_min, omega_max, modes+1)
    omega = np.sqrt(edges[:-1]*edges[1:])
    occupation = 1/np.expm1(beta*omega)
    g = np.sqrt((sigma**2/modes)/(2*occupation+1))
    return omega, g, occupation


def heom(pulses=None, depth=5, reset=False, initial=None, **bath_parameters):
    """Keep every ADO across pulse boundaries unless reset=True (ablation)."""
    import qutip as qt
    from qutip.solver.heom import BosonicBath, HEOMSolver
    pulses = quantum_schedule() if pulses is None else pulses
    omega, g, nb = mode_parameters(**bath_parameters)
    if depth < 1 or comb(2*len(omega)+depth, depth) > 50000:
        raise ValueError("Invalid/too-large hierarchy; reduce modes or depth")
    cr, vr, ci, vi = [], [], [], []
    for w, coupling, nbar in zip(omega, g, nb):
        variance = coupling**2*(2*nbar+1)
        # C(t)=g^2[(n+1)e^{-iwt}+n e^{iwt}].
        # QuTiP represents Re C and Im C separately, then combines matching
        # exponents. Complex coefficients occur because cos/sin use exp pairs.
        cr.extend([variance/2, variance/2])
        vr.extend([1j*w, -1j*w])
        ci.extend([-0.5j*coupling**2, 0.5j*coupling**2])
        vi.extend([1j*w, -1j*w])
    bath = BosonicBath(qt.Qobj(Z/2), cr, vr, ci, vi, combine=True)
    rho0 = qt.ket2dm(qt.basis(2,0)) if initial is None else qt.Qobj(initial)
    state = rho0
    states, ado_count = [rho0.full()], None
    for p in pulses:
        solver = HEOMSolver(qt.Qobj(p.h), bath, max_depth=depth,
                            options=dict(store_states=True, store_ados=True,
                                         progress_bar="", atol=1e-10, rtol=1e-9,
                                         nsteps=20000, method="adams"))
        ado_count = len(solver.ados.labels)
        result = solver.run(state, [0., p.duration])
        rho = result.states[-1]
        states.append(rho.full())
        # Passing rho alone here would discard the very memory we study.
        state = rho if reset else result.ado_states[-1]
    return np.asarray(states), dict(ado_count=ado_count, depth=depth,
                                   mode_frequencies=omega.tolist(),
                                   couplings=g.tolist(), occupations=nb.tolist())


def explicit_bath(pulses=None, fock=6, initial=None, **bath_parameters):
    """Independent dense evolution of system + finite harmonic oscillators.

Each oscillator starts in a normalized truncated Gibbs state. Both the Fock
tail and displacement-induced occupation require convergence checks.
"""
    import qutip as qt
    pulses = quantum_schedule() if pulses is None else pulses
    omega, g, nb = mode_parameters(**bath_parameters)
    if fock < 2 or 2*fock**len(omega) > 2048:
        raise ValueError("Explicit bath dimension too large or invalid Fock cutoff")
    factors = [qt.qeye(2)] + [qt.qeye(fock) for _ in omega]
    def local(op, index):
        terms = factors.copy()
        terms[index] = op
        return qt.tensor(terms)
    rho_s = qt.ket2dm(qt.basis(2,0)) if initial is None else qt.Qobj(initial)
    thermal = [qt.thermal_dm(fock, float(n)) for n in nb]
    rho = qt.tensor([rho_s] + thermal).full()
    z = local(qt.Qobj(Z/2), 0)
    hb = 0 * z
    for j, (w, coupling) in enumerate(zip(omega, g)):
        a = local(qt.destroy(fock), j+1)
        hb += w*a.dag()*a + coupling*z*(a+a.dag())
    states = [rho_s.full()]
    for p in pulses:
        h = (hb + local(qt.Qobj(p.h), 0)).full()
        e, v = np.linalg.eigh(h)
        transformed = v.conj().T @ rho @ v
        transformed *= np.exp(-1j*p.duration*(e[:,None]-e[None,:]))
        rho = v @ transformed @ v.conj().T
        total = qt.Qobj(rho, dims=[[2]+[fock]*len(omega)]*2)
        states.append(total.ptrace(0).full())
    tails = (nb/(nb+1))**fock
    return np.asarray(states), dict(fock=fock, joint_dimension=len(rho),
                                   initial_thermal_tail_per_mode=tails.tolist())
