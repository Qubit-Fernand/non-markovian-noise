"""Decomposition into the pulse-level native set {r, rz, sqisw}.

Every single-qubit unitary is ONE square XY pulse followed by an ideal Z:

    U ~ Rz(a) Rx(t) Rz(b) = Rz(a+b) R_{phi=-b}(t),   r = R_phi(t).

A ZZ rotation needs two exchange gates (S = sqisw = exp(-i pi/8 (XX+YY))):

    exp(-i theta ZZ/2) ~ (k3 x k4) S (Rx(alpha) x Ry(pi)) S (k1 x k2),
    sin((pi-alpha)/2) = sqrt(2) sin(|theta|/2),      |theta| <= pi/2,

where ~ is equality up to a global phase. The interleaved pair fixes the
nonlocal content; the outer locals k1..k4 are solved numerically once per
angle and checked to machine precision.
"""
from functools import lru_cache

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

from .circuit import SQISW, CircuitLayers, Gate, Layer, ParallelBlock
from .pauli import I, X, Y, Z

_TOL = 1e-9


def _rot(axis, angle):
    return expm(-.5j*angle*axis)


def _su2(params):
    """Rz(a) Rx(t) Rz(b) in closed form."""
    a, t, b = params
    c, sn = np.cos(t/2), -1j*np.sin(t/2)
    return np.array([[c*np.exp(-.5j*(a+b)), sn*np.exp(-.5j*(a-b))],
                     [sn*np.exp(.5j*(a-b)), c*np.exp(.5j*(a+b))]])


def equal_up_to_phase(u, v, atol=1e-9):
    return abs(abs(np.trace(u.conj().T@v))-len(u)) < atol*len(u)


def zxz(unitary):
    """Angles (a, t, b) with unitary ~ Rz(a) Rx(t) Rz(b), 0 <= t <= pi."""
    u = np.asarray(unitary, complex)
    u = u/np.sqrt(np.linalg.det(u))
    t = 2*np.arctan2(abs(u[0, 1]), abs(u[0, 0]))
    if abs(u[0, 1]) < 1e-12:      # diagonal: a pure Z rotation
        a, b = -2*np.angle(u[0, 0]), 0.
    elif abs(u[0, 0]) < 1e-12:    # anti-diagonal: only a-b is defined
        a, b = -2*(np.angle(u[0, 1])+np.pi/2), 0.
    else:
        total, difference = -2*np.angle(u[0, 0]), -2*(np.angle(u[0, 1])+np.pi/2)
        a, b = (total+difference)/2, (total-difference)/2
    if not equal_up_to_phase(_su2((a, t, b)), u):
        raise ArithmeticError("ZXZ extraction failed")
    return float(a), float(t), float(b)


def _wrap(angle):
    return float((angle+np.pi) % (2*np.pi)-np.pi)


def pulse_gates(unitary, qubit):
    """[r pulse], [rz] realising a single-qubit unitary; empty lists if trivial."""
    a, t, b = zxz(unitary)
    pulse = [Gate("r", (qubit,), t, _wrap(-b))] if t > _TOL else []
    z = _wrap(a+b)
    return pulse, ([Gate("rz", (qubit,), z)] if abs(z) > _TOL else [])


@lru_cache(maxsize=64)
def zz_locals(theta):
    """(k1, k2, m1, m2, k3, k4) of the two-sqisw form of exp(-i theta ZZ/2)."""
    if not np.isfinite(theta) or abs(theta) > np.pi/2+1e-12:
        raise ValueError("two sqisw gates reach |theta| <= pi/2 only")
    alpha = np.pi-2*np.arcsin(min(1., np.sqrt(2)*np.sin(abs(theta)/2)))
    m1, m2 = _rot(X, alpha), _rot(Y, np.pi)
    core = SQISW@np.kron(m1, m2)@SQISW
    target = expm(-.5j*theta*np.kron(Z, Z))

    def residual(p):
        trial = np.kron(_su2(p[6:9]), _su2(p[9:12]))@core@np.kron(_su2(p[0:3]), _su2(p[3:6]))
        difference = (trial-np.exp(1j*p[12])*target).ravel()
        return np.concatenate((difference.real, difference.imag))

    rng = np.random.default_rng(20261114)
    for _ in range(200):
        result = least_squares(residual, rng.uniform(-np.pi, np.pi, 13),
                               xtol=1e-15, ftol=1e-15, gtol=1e-15)
        if result.cost < 1e-24:
            p = result.x
            out = (_su2(p[0:3]), _su2(p[3:6]), m1, m2, _su2(p[6:9]), _su2(p[9:12]))
            total = np.kron(out[4], out[5])@core@np.kron(out[0], out[1])
            if equal_up_to_phase(total, target, 1e-11):
                return out
    raise ArithmeticError("no local gates found for the sqisw decomposition")


def _one_qubit_blocks(unitaries):
    """Parallel r block and rz block for {qubit: 2x2 unitary}; skips identities."""
    pulses, zs = [], []
    for qubit, u in sorted(unitaries.items()):
        p, z = pulse_gates(u, qubit)
        pulses += p
        zs += z
    return [ParallelBlock(tuple(b)) for b in (pulses, zs) if b]


def ising_sqisw_layers(n_qubits=4, *, theta_x=.5, theta_z=-.075, steps=1):
    """First-order Trotter step of the open Ising chain in {r, rz, sqisw}.

    Logical circuit (the HTML proposal): Rx(theta_x) on every qubit, then
    exp(-i theta_z ZZ/2) on the even edges (0,1),(2,3),..., then on the odd
    edges. Adjacent single-qubit unitaries are merged before pulses are made.

    One learning layer per exchange-gate round: the single-qubit pulses that
    precede a round belong to its layer and the final pulses to the last one,
    so a step on n >= 3 qubits has four layers:
        L1 pulses, S(even)   L2 pulses, S(even)
        L3 pulses, S(odd)    L4 pulses, S(odd), pulses
    """
    if n_qubits < 2 or steps < 1:
        raise ValueError("Ising chain needs at least two qubits and one step")
    k1, k2, m1, m2, k3, k4 = zz_locals(float(theta_z))
    rounds = [tuple((q, q+1) for q in range(0, n_qubits-1, 2))]
    if n_qubits > 2:
        rounds.append(tuple((q, q+1) for q in range(1, n_qubits-1, 2)))
    eye = np.eye(2, dtype=complex)
    pending = {q: eye for q in range(n_qubits)}  # unitaries waiting for a pulse
    layers = []
    for step in range(steps):
        for q in range(n_qubits):
            pending[q] = _rot(X, theta_x)@pending[q]
        for parity, edges in zip(("even", "odd"), rounds):
            exchange = ParallelBlock(tuple(Gate("sqisw", e) for e in edges))
            for a, b in edges:
                pending[a], pending[b] = k1@pending[a], k2@pending[b]
            layers.append(Layer(f"s{step}-{parity}-1", (*_one_qubit_blocks(pending), exchange), step))
            pending = {q: eye for q in range(n_qubits)}
            for a, b in edges:
                pending[a], pending[b] = m1, m2
            blocks = [*_one_qubit_blocks(pending), exchange]
            pending = {q: eye for q in range(n_qubits)}
            for a, b in edges:
                pending[a], pending[b] = k3, k4
            last = step == steps-1 and edges is rounds[-1]
            if last:
                blocks += _one_qubit_blocks(pending)
            layers.append(Layer(f"s{step}-{parity}-2", tuple(blocks), step))
    return CircuitLayers(n_qubits, tuple(layers), "ising-sqisw")


def ising_trotter_unitary(n_qubits, theta_x, theta_z, steps=1):
    """Reference target built directly from the Hamiltonian terms."""
    from .pauli import embed
    xs = sum(embed(X, (q,), n_qubits) for q in range(n_qubits))
    zz = lambda edges: sum(embed(np.kron(Z, Z), e, n_qubits) for e in edges)
    even = [(q, q+1) for q in range(0, n_qubits-1, 2)]
    odd = [(q, q+1) for q in range(1, n_qubits-1, 2)]
    step = expm(-.5j*theta_x*xs)
    for edges in (even, odd):
        if edges:
            step = expm(-.5j*theta_z*zz(edges))@step
    return np.linalg.matrix_power(step, steps)
