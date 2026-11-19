"""Checks of the physical bridge, identifiability and implementable PEC."""
import numpy as np
import pytest
from nmnoise.core import X, Y, Z, trace_distance
from nmnoise.quantum import heom
from nmnoise.core import Pulse
from nmnoise.spp import (ProcessConfig, CALIBRATION, heom_probabilities,
    explicit_probabilities, heom_solver, initial_ados, evolve_ados,
    fit_terminal_counts, independent_projection, inverse_qp, convolution_matrix,
    ising_circuits, history_states, twirled_heom_state, pec_statistics)


def test_ado_restart_preserves_complex_coherences_and_memory():
    config = ProcessConfig()
    solver = heom_solver(config, depth=8)
    ket = np.array([1, 1j])/np.sqrt(2)
    rho = np.outer(ket, ket.conj())
    ados = initial_ados(solver, rho)
    first = evolve_ados(solver, ados, config.duration)
    second = evolve_ados(solver, first, config.duration)
    whole = evolve_ados(solver, ados, 2*config.duration)
    np.testing.assert_allclose(second, whole, atol=2e-8)
    reference, _ = heom([Pulse('idle', 24., (0,), np.zeros((2, 2)))],
                        depth=8, initial=rho, **config.bath())
    np.testing.assert_allclose(second[0], reference[-1], atol=2e-8)


def test_heom_process_matches_independent_oscillator_and_has_memory():
    p, _ = heom_probabilities(depth=8)
    exact, _ = explicit_probabilities(fock=12)
    np.testing.assert_allclose(p, exact, atol=2e-7)
    assert abs(p.sum()-1) < 1e-9 and p.min() >= 0
    assert np.max(np.abs(p-independent_projection(p))) > 1e-3
    p0, _ = heom_probabilities(ProcessConfig(sigma=0), depth=4)
    np.testing.assert_allclose(p0, [1, 0, 0, 0], atol=1e-10)


def test_terminal_only_identifiability_and_boundary_fit():
    p = np.array([.75, .06, .08, .11])
    assert np.linalg.matrix_rank(np.vstack([np.ones(4), CALIBRATION])) == 4
    fitted, _ = fit_terminal_counts((1-CALIBRATION @ p)*50000, 100000)
    np.testing.assert_allclose(p, fitted, atol=1e-12)
    bounded, info = fit_terminal_counts([0, 0, 10], 100)
    assert info['success'] and bounded.min() >= 0
    np.testing.assert_allclose(bounded.sum(), 1, atol=1e-9)
    ind = independent_projection(p)
    recovered, _ = fit_terminal_counts((1-CALIBRATION @ ind)*50000,
                                       100000, independent=True)
    np.testing.assert_allclose(recovered, ind, atol=1e-6)


def test_history_inverse_cancels_for_noncommuting_ising_controls():
    p = np.array([.75, .06, .08, .11])
    q, info = inverse_qp(p)
    np.testing.assert_allclose(convolution_matrix(p) @ q, [1, 0, 0, 0], atol=1e-10)
    circuits, obs, _ = ising_circuits(count=1)
    states = history_states(circuits[0])
    mitigated = np.einsum('h,hij->ij', convolution_matrix(p) @ q, states)
    np.testing.assert_allclose(mitigated, states[0], atol=1e-10)
    assert info['gamma'] > 1
    with pytest.raises(ValueError, match='singular'):
        inverse_qp(np.ones(4)/4)
    response = np.einsum('ij,hji->h', obs[0], states).real
    rng = np.random.default_rng(11)
    samples = [pec_statistics(response, p, q, 1000, rng) for _ in range(300)]
    estimates = np.array([s[0] for s in samples])
    variance = samples[0][2]
    assert abs(estimates.mean()-response[0]) < 5*np.sqrt(variance/len(samples))
    assert .7 < estimates.var(ddof=1)/variance < 1.3


def test_direct_four_qubit_twirl_and_physical_corrections():
    p, _ = heom_probabilities(depth=8)
    circuits, _, _ = ising_circuits(count=1)
    states = history_states(circuits[0])
    # Both slots corrected: validates unchanged bath and XOR order.
    direct = twirled_heom_state(circuits[0], depth=8, correction=3)
    predicted = np.einsum('h,hij->ij', p[np.arange(4)^3], states)
    assert trace_distance(direct, predicted) < 3e-7
    assert abs(np.trace(direct)-1) < 1e-8
    assert np.linalg.eigvalsh(direct).min() > -1e-8
