"""Finite-pulse slots and TensorCircuit gate-after response evaluation."""
from itertools import product

import numpy as np
import tensorcircuit as tc

from .heom import Segment, Slot
from .pauli import I, X, Z, paulis


def embed(operator, sites, n):
    """Small-system reference embedding; site 0 is the left tensor factor."""
    sites = tuple(sites)
    if len(set(sites)) != len(sites) or any(s < 0 or s >= n for s in sites):
        raise ValueError("invalid sites")
    operator = np.asarray(operator)
    if operator.shape != (2**len(sites),)*2:
        raise ValueError("operator size does not match sites")
    result = np.zeros((2**n, 2**n), complex)
    for column in range(2**n):
        bits = [(column >> (n-1-j)) & 1 for j in range(n)]
        local = sum(bits[s] << (len(sites)-1-j) for j, s in enumerate(sites))
        for target in range(2**len(sites)):
            out = bits.copy()
            for j, site in enumerate(sites):
                out[site] = (target >> (len(sites)-1-j)) & 1
            row = sum(b << (n-1-j) for j, b in enumerate(out))
            result[row, column] = operator[target, local]
    return result


def ising_slots(theta_x=.5, theta_z=-.075, rotation_duration=1., cx_duration=2., steps=1):
    """Four qubits, HTML table convention: middle Rz belongs to slot THREE.

    Disjoint CX gates are parallel. Within a slot rotations and CX pulses
    are sequential. Time units are arbitrary but must match sigma/gamma.
    """
    if steps < 1 or rotation_duration <= 0 or cx_duration <= 0:
        raise ValueError("steps/durations must be positive")
    n = 4
    rx = sum(embed(X, (j,), n) for j in range(n))*theta_x/(2*rotation_duration)
    rz_parallel = sum(embed(Z, (j,), n) for j in (1, 3))*theta_z/(2*rotation_duration)
    rz_middle = embed(Z, (2,), n)*theta_z/(2*rotation_duration)
    hc = np.pi/cx_duration*np.kron((I-Z)/2, (I-X)/2)
    cx_parallel = embed(hc, (0, 1), n)+embed(hc, (2, 3), n)
    cx_middle = embed(hc, (1, 2), n)
    seg = Segment
    one_step = (
        Slot((seg(rx, rotation_duration), seg(cx_parallel, cx_duration)), "rx-cx-even"),
        Slot((seg(rz_parallel, rotation_duration), seg(cx_parallel, cx_duration)), "rz-cx-even"),
        Slot((seg(cx_middle, cx_duration), seg(rz_middle, rotation_duration)), "cx-rz-middle"),
        Slot((seg(cx_middle, cx_duration),), "cx-middle"),
    )
    return one_step*steps


def history_expectation(slots, history, observable, initial=None):
    """TensorCircuit state TN; error P_a is inserted AFTER each ideal U_j.

    Ideal layers are currently supplied as dense small-system matrices. This
    evaluator does not model noisy corrections or physical twirl pulses.
    """
    if not slots or len(history) != len(slots):
        raise ValueError("one label per slot required")
    d = len(slots[0].ideal)
    n = d.bit_length()-1
    labels = tuple(history)
    if any(not isinstance(a, (int, np.integer)) or not 0 <= a < 4**n for a in labels):
        raise ValueError("invalid Pauli history")
    if initial is not None:
        initial = np.asarray(initial, complex)
        if initial.shape != (d,) or not np.isclose(np.linalg.norm(initial), 1):
            raise ValueError("initial pure state must be normalized")
    observable = np.asarray(observable, complex)
    if observable.shape != (d, d) or not np.allclose(observable, observable.conj().T):
        raise ValueError("observable must be Hermitian")
    with tc.runtime_backend("numpy"), tc.runtime_dtype("complex128"):
        c = tc.Circuit(n, inputs=initial)
        for slot, label in zip(slots, labels):
            c.any(*range(n), unitary=slot.ideal)
            for site in range(n):
                digit = (label >> (2*(n-1-site))) & 3
                if digit:
                    c.any(site, unitary=paulis(1)[digit])
        return float(np.real(c.expectation((observable, list(range(n))))))


def response_tensor(slots, observable, initial=None, max_entries=65536):
    n = len(slots[0].ideal).bit_length()-1
    m = 4**n
    if m**len(slots) > max_entries:
        raise ValueError("response enumeration exceeds max_entries; use sampled histories")
    return np.array([history_expectation(slots, h, observable, initial)
                     for h in product(range(m), repeat=len(slots))]).reshape((m,)*len(slots))
