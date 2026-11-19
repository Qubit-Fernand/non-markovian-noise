"""Conventions: hbar=1, time in ns, angular frequencies in rad/ns.

Qubit zero is the leftmost / most significant tensor factor. All controls have
finite duration. The example CNOT Hamiltonian is an engineered ideal control,
not a calibrated superconducting cross-resonance pulse.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)


@dataclass(frozen=True)
class Pulse:
    label: str
    duration: float
    sites: tuple[int, ...]
    h: np.ndarray


def embed(op, sites, n):
    """Embed an ordered, contiguous one- or two-site operator."""
    if len(sites) not in (1, 2) or tuple(sites) != tuple(range(sites[0], sites[0] + len(sites))):
        raise ValueError("Only ordered contiguous 1/2-site terms are supported")
    if sites[0] < 0 or sites[-1] >= n:
        raise ValueError("Site outside system")
    return np.kron(np.kron(np.eye(2**sites[0]), op), np.eye(2**(n - sites[-1] - 1)))


def ghz_schedule(n=3):
    if n < 2:
        raise ValueError("GHZ demo needs at least two qubits")
    # exp(-i*pi/4*Y)|0> = |+>
    pulses = [Pulse("Ry(pi/2)", 20., (0,), np.pi / 80 * Y)]
    p1 = (I - Z) / 2
    # exp[-i*pi*P1 (I-X)/2] = controlled-X, without relative phases.
    for q in range(n - 1):
        pulses.append(Pulse(f"CX({q},{q+1})", 80., (q, q+1), np.pi / 80 * np.kron(p1, (I-X)/2)))
    pulses.append(Pulse("idle", 20., (0,), np.zeros((2, 2), complex)))
    return pulses


def timeline(pulses, dt):
    if dt <= 0:
        raise ValueError("dt must be positive")
    indices = []
    for k, pulse in enumerate(pulses):
        steps = round(pulse.duration / dt)
        if steps <= 0 or not np.isclose(steps * dt, pulse.duration):
            raise ValueError("Each pulse duration must be an integer multiple of dt")
        indices.extend([k] * steps)
    return np.asarray(indices)


def ket0(n):
    psi = np.zeros(2**n, complex)
    psi[0] = 1
    return psi


def ideal_state(pulses, n, initial=None):
    psi = ket0(n) if initial is None else np.asarray(initial, complex).copy()
    for pulse in pulses:
        psi = expm(-1j * pulse.duration * embed(pulse.h, pulse.sites, n)) @ psi
    return psi


def ensemble_rho(states):
    # Each row contains one pure state. This is E[|psi><psi|], not |E[psi]><...|.
    return np.einsum("bi,bj->ij", states, states.conj()) / len(states)


def trace_distance(a, b):
    delta = (a - b + (a-b).conj().T) / 2
    return float(np.abs(np.linalg.eigvalsh(delta)).sum() / 2)


def state_summary(rho, target):
    n = int(round(np.log2(len(rho))))
    xxx = X
    for _ in range(n-1):
        xxx = np.kron(xxx, X)
    return dict(fidelity=float(np.vdot(target, rho @ target).real),
                purity=float(np.trace(rho @ rho).real),
                trace_real=float(np.trace(rho).real),
                trace_imag=float(np.trace(rho).imag),
                min_eigenvalue=float(np.linalg.eigvalsh((rho+rho.conj().T)/2).min()),
                probabilities=np.diag(rho).real.tolist(),
                x_parity=float(np.trace(rho @ xxx).real),
                ghz_coherence_real=float(rho[0, -1].real),
                ghz_coherence_imag=float(rho[0, -1].imag))
