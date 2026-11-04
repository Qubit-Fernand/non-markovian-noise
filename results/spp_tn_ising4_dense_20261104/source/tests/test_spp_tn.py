"""Independent small-system checks for the time/hierarchy tensor networks."""
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

pytest.importorskip("tensorcircuit", reason="install the tn-ng or tn-legacy extra")

from spp_tn import MPS, Truncation, characters, paulis, walsh
from spp_tn.circuits import history_expectation, ising_slots, response_tensor, embed
from spp_tn.heom import DenseHermiteHEOM, HermiteHEOM, Mode, Segment, Slot
from spp_tn.mitigation import (amplify, inverse_als, inverse_dense, real_tensor,
                               sample_corrections, quasiprobability_report)
from spp_tn.pauli import I, X, Y, Z
from spp_tn.process import build_spp, independent_model, probability_report, sample_histories


def correlated_process(length=3):
    return MPS.product([np.array([.94, .01, .02, .03])]*length).scaled(.65).add(
        MPS.product([np.array([.72, .06, .08, .14])]*length).scaled(.35))


def test_mps_algebra_and_rounding(tmp_path):
    rng = np.random.default_rng(3)
    a = rng.normal(size=(3, 4, 2))+1j*rng.normal(size=(3, 4, 2))
    b = rng.normal(size=a.shape)
    x, y = MPS.from_dense(a), MPS.from_dense(b)
    assert_allclose(x.dense(), a, atol=1e-13)
    assert_allclose(x.add(y).dense(), a+b, atol=1e-13)
    assert_allclose(x.hadamard(y).dense(), a*b, atol=1e-13)
    assert_allclose(x.inner(y), np.vdot(a, b), atol=1e-12)
    assert_allclose(x.total(), a.sum(), atol=1e-12)
    assert_allclose(real_tensor(x).dense(), a.real, atol=1e-12)
    rounded, errors = x.round(Truncation(max_bond=1))
    assert np.linalg.norm(rounded.dense()-a) <= np.linalg.norm(errors)+1e-12
    assert_allclose(x.round()[0].dense(), a, atol=1e-12)
    path = tmp_path/"mps.npz"
    x.save(path)
    assert_allclose(MPS.load(path).dense(), a, atol=1e-12)
    with pytest.raises(FileExistsError):
        x.save(path)
    with pytest.raises(ValueError, match="enumeration"):
        x.dense(max_entries=1)


@pytest.mark.parametrize("n", [1, 2])
def test_pauli_characters_against_matrices(n):
    ps, w = paulis(n), characters(n)
    assert_allclose(w @ w.T, 4**n*np.eye(4**n))
    for a, b in product(range(4**n), repeat=2):
        assert_allclose(ps[a]@ps[b]@ps[a], w[b, a]*ps[b])
        # phase-free group law is XOR even with I,X,Y,Z label order
        assert_allclose(abs(np.trace(ps[a ^ b].conj().T @ ps[a]@ps[b])), 2**n)


def test_walsh_convolution_and_gain_zero():
    p = correlated_process()
    dense = p.dense()
    assert_allclose(walsh(walsh(p, 1), 1, inverse=True).dense(), dense, atol=1e-14)
    p2, _ = amplify(p, 1, 2)
    expected = np.zeros_like(dense)
    for a in product(range(4), repeat=3):
        for b in product(range(4), repeat=3):
            expected[tuple(x ^ y for x, y in zip(a, b))] += dense[a]*dense[b]
    assert_allclose(p2.dense(), expected, atol=1e-13)
    identity, _ = amplify(p, 1, 0)
    delta = np.zeros_like(dense)
    delta[0, 0, 0] = 1
    assert_allclose(identity.dense(), delta, atol=1e-14)


def test_marginals_sampling_and_rejection():
    p = correlated_process(2)
    ip = independent_model(p).dense()
    pd = p.dense().real
    assert_allclose(ip, np.outer(pd.sum(axis=1), pd.sum(axis=0)), atol=1e-14)
    histories, audit = sample_histories(p, 10000, np.random.default_rng(4))
    histogram = np.zeros((4, 4))
    np.add.at(histogram, tuple(histories.T), 1/len(histories))
    assert np.max(np.abs(histogram-pd)) < .015
    assert audit["clipped_conditional_mass_sum"] < 1e-10
    signed = MPS.product([np.array([1.1, -.1, 0, 0])])
    with pytest.raises(ValueError, match="verified"):
        sample_histories(signed, 1, np.random.default_rng())
    with pytest.raises(ValueError, match="verified"):
        sample_histories(p, 1, np.random.default_rng(), max_entries=1)
    assert probability_report(p, max_entries=1)["valid"] is None


def test_hierarchy_mps_matches_dense_full_ados():
    modes = [Mode(Z/2, .3, .2, 3), Mode(X/2, .2, 0, 3)]
    slot = Slot((Segment(.4*Y, .2), Segment(.3*X+.1*Z, .15)))
    engine = HermiteHEOM(modes, max_dt=.025, truncation=Truncation(cutoff=1e-13))
    dense = DenseHermiteHEOM(modes)
    rho = np.array([[.5, .5], [.5, .5]], complex)
    state = MPS([rho.reshape(1, 4, 1, order="F"), *engine.root().cores])
    actual = engine.propagate(state, slot).dense()
    expected = dense.propagate(state, slot).dense()
    assert_allclose(actual, expected, atol=2e-9)
    assert engine.stats.max_bond_seen > 1


def test_gate_after_probe_convention_noncommuting():
    slot = Slot((Segment(.7*X, .4), Segment(.3*Y, .3)))
    engine = DenseHermiteHEOM([Mode(Z/2, .4, .1, 4)])
    # Direct extended map G followed on its input by ideal inverse U^-1.
    g = np.eye(16, dtype=complex)
    for segment in slot.segments:
        g = expm(segment.duration*engine.generator(segment).toarray()) @ g
    u = slot.ideal
    for p in paulis(1):
        expected_input = np.zeros((4, 4), complex)
        expected_input[:, 0] = (u.conj().T@p@u).reshape(-1, order="F")
        output = (g@expected_input.reshape(-1)).reshape(4, 4)
        expected = p.reshape(-1, order="F").conj()@output/2
        assert_allclose(engine.response(slot, p, engine.root()).dense(), expected, atol=1e-12)


def test_static_gaussian_spp_against_quadrature_and_temporal_correlation():
    sigma, duration, length = .6, .7, 3
    modes = [Mode(Z/2, sigma, 0, 9)]
    slots = [Slot((Segment(np.zeros((2, 2)), duration),))]*length
    engine = DenseHermiteHEOM(modes)
    p, records = build_spp(engine, slots, time_truncation=Truncation(cutoff=1e-13))
    # Independent exact integration over a common STATIC Gaussian amplitude.
    nodes, weights = np.polynomial.hermite.hermgauss(50)
    prob_z = np.sin(sigma*duration*np.sqrt(2)*nodes/2)**2
    expected = np.zeros((4,)*length)
    for h in product((0, 3), repeat=length):
        integrand = np.ones(len(nodes))
        for label in h:
            integrand *= prob_z if label == 3 else 1-prob_z
        expected[h] = weights@integrand/np.sqrt(np.pi)
    assert_allclose(p.dense(), expected, atol=2e-9)
    assert records[1]["probes"] < 4**3
    assert np.linalg.norm(p.dense()-independent_model(p).dense()) > .005
    assert probability_report(p, tolerance=1e-8)["valid"]


def test_mps_spp_vs_dense_and_no_noise():
    slots = [Slot((Segment(.5*X, .2),)), Slot((Segment(.3*Z, .15),))]
    modes = [Mode(Z/2, .4, .2, 4)]
    p, _ = build_spp(HermiteHEOM(modes, max_dt=.025, truncation=Truncation(cutoff=1e-13)),
                     slots, time_truncation=Truncation(cutoff=1e-13))
    expected, _ = build_spp(DenseHermiteHEOM(modes), slots)
    assert_allclose(p.dense(), expected.dense(), atol=1e-9)
    no_noise, _ = build_spp(DenseHermiteHEOM([Mode(Z/2, 0, .1, 2)]), slots)
    delta = np.zeros((4, 4))
    delta[0, 0] = 1
    assert_allclose(no_noise.dense(), delta, atol=1e-12)


def test_time_truncation_keeps_weighted_memory_singular_vectors():
    """A left-canonical tag alone has unit singular values: splitting it would
    select an arbitrary subspace and lose the physical Schmidt weights.
    Compare the actual truncated construction to an independent dense split.
    """
    slots = [Slot((Segment(.5*X, .5),)), Slot((Segment(.3*Z, .4),))]
    engine = DenseHermiteHEOM([Mode(Z/2, .6, .2, 5)])
    responses = np.array([engine.response(slots[0], p, engine.root()).dense() for p in paulis(1)])
    branches = characters(1)@responses/4
    u, s, vh = np.linalg.svd(branches, full_matrices=False)
    memory = MPS.product([s[0]*vh[0]])
    second = np.array([engine.response(slots[1], p, memory).value([0]) for p in paulis(1)])
    expected = np.outer(u[:, 0], characters(1)@second/4)
    truncated, records = build_spp(engine, slots, time_truncation=Truncation(max_bond=1))
    assert_allclose(truncated.dense(), expected, atol=1e-12)
    assert_allclose(records[0]["time_discarded_norm"], np.linalg.norm(s[1:]), atol=1e-12)


def test_direct_multitime_twirl_matches_spp_circuit_response():
    slots = [Slot((Segment(.4*X, .3),)), Slot((Segment(.2*Y, .25),))]
    engine = DenseHermiteHEOM([Mode(Z/2, .5, .2, 4)])
    p, _ = build_spp(engine, slots)
    initial = np.array([1, 1j])/np.sqrt(2)
    rho = np.outer(initial, initial.conj())
    state = np.zeros((4, 4), complex)
    state[:, 0] = rho.reshape(-1, order="F")
    def conjugate_ados(data, matrix):
        result = np.empty_like(data)
        for k in range(4):
            result[:, k] = (matrix@data[:, k].reshape(2, 2, order="F")
                            @matrix.conj().T).reshape(-1, order="F")
        return result
    for slot in slots:
        averaged = np.zeros_like(state)
        u = slot.ideal
        for frame in paulis(1):
            # Gate-after twirl: frame before physical gate is U† P U.
            branch = conjugate_ados(state, u.conj().T@frame@u)
            for segment in slot.segments:
                branch = expm_multiply(segment.duration*engine.generator(segment),
                                       branch.reshape(-1)).reshape(4, 4)
            averaged += conjugate_ados(branch, frame)/4
        state = averaged
    direct = np.trace(X@state[:, 0].reshape(2, 2, order="F")).real
    response = response_tensor(slots, X, initial)
    assert_allclose(np.sum(p.dense()*response), direct, atol=1e-11)


def test_inverse_als_matches_dense_and_pec_weights():
    p = correlated_process()
    q, report = inverse_als(p, 1, bond=4, sweeps=10, tolerance=1e-7)
    exact = inverse_dense(p, 1)
    assert_allclose(q.dense(), exact.dense(), atol=2e-7)
    assert abs(q.total()-1) < 1e-9
    assert report["history"][-1]["residual_mse"] < 1e-12
    correction, weights, cost = sample_corrections(q, 2000, np.random.default_rng(5))
    assert correction.shape == (2000, 3)
    assert_allclose(np.abs(weights), cost["gamma"])
    assert cost["gamma"] > 1
    with pytest.raises(ValueError, match="singular"):
        inverse_dense(MPS.product([np.ones(4)/4]), 1)


def test_ising_slots_and_tc_error_order():
    slots = ising_slots()
    actual = np.eye(16, dtype=complex)
    for slot in slots:
        actual = slot.ideal@actual
    zz = sum(embed(np.kron(Z, Z), (j, j+1), 4) for j in range(3))
    xx = sum(embed(X, (j,), 4) for j in range(4))
    expected = expm(-1j*(-.075)/2*zz)@expm(-1j*.5/2*xx)
    assert_allclose(actual, expected, atol=1e-13)
    slot = Slot((Segment(np.pi/4*Y, 1),))
    # Ry(pi/2)|0> = |+>; a Z error AFTER it changes X from +1 to -1.
    assert_allclose(history_expectation([slot], [3], X), -1, atol=1e-12)


def test_large_history_tn_path_does_not_enumerate():
    p = correlated_process(40)
    assert p.storage < 1000
    assert_allclose(p.total(), 1, atol=1e-12)
    p2, _ = amplify(p, 1, 2, Truncation(max_bond=4, cutoff=1e-13))
    assert_allclose(p2.total(), 1, atol=1e-10)
    assert p2.storage < 3000
    with pytest.raises(ValueError, match="enumeration"):
        quasiprobability_report(p2)
