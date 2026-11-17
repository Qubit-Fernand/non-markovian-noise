"""Complex baths, stage-dependent coupling, virtual gates, logical ideal gates."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm
from heom_spp.bath import (Mode, White, classical_mode, load_exponentials,
                           quantum_modes)
from heom_spp.circuit import (CircuitLayers, Gate, Layer, ParallelBlock,
                              gate_unitary, layer_unitary)
from heom_spp.heom import DenseHEOM, TDVPHEOM
from heom_spp.mps import MPS, Truncation, split
from heom_spp.pauli import I, X, Y, Z, embed, paulis
from heom_spp.pulse import (Frame, PulseDurations, Segment, Slot,
                            compile_layers, gate_hamiltonian)
from heom_spp.spp import build_spp

P0 = (I+Z)/2
PLUS = np.full((2, 2), .5, complex)


def root_matrix(engine, rho, slot):
    state = MPS([rho.reshape(1, -1, 1, order="F"), *engine.root().cores])
    final = engine.propagate(state, slot).dense()
    d = len(rho)
    return final.reshape(d*d, -1)[:, 0].reshape(d, d, order="F")


def idle(duration, dim=2, **options):
    return Slot((Segment(np.zeros((dim, dim)), duration, **options),))


def test_split_cutoff_and_rank_semantics():
    matrix = np.diag([3., 2., 1e-3, 0.])
    u, s, vh, lost = split(matrix)
    assert len(s) == 3 and lost == 0  # exact zeros are dropped, nothing else
    u, s, vh, lost = split(matrix, Truncation(cutoff=2e-3))
    assert len(s) == 2 and np.isclose(lost, 1e-3)
    u, s, vh, lost = split(matrix, Truncation(max_bond=1))
    assert len(s) == 1 and np.isclose(lost, np.hypot(2, 1e-3))
    assert_allclose((u*s)@vh, np.diag([3., 0, 0, 0]), atol=1e-14)
    assert len(split(np.zeros((2, 3)))[1]) == 1


@pytest.mark.parametrize("terms", [
    ([.6+.3j], [.05+.02j]),
    ([.6+.3j, .15-.4j], [.05+.02j, .02-.03j]),
])
def test_complex_bath_exact_pure_dephasing(terms):
    """Q = |0><0|: rho_01(t) = rho_01(0) exp(-g2(t)), g2 the double integral
    of the COMPLEX C(t). Its phase only appears if left/right nodes carry the
    correct, conjugated coefficients.
    """
    w, g = (np.array(x) for x in terms)
    duration = 1.5
    engine = DenseHEOM(quantum_modes(P0, w, g, 7), max_entries=10**7)
    rho = root_matrix(engine, PLUS, idle(duration))
    g2 = np.sum(g*(np.exp(-w*duration)-1+w*duration)/w**2)
    assert abs(g2.imag) > 1e-3
    assert_allclose(rho[0, 1], .5*np.exp(-g2), atol=5e-9)
    assert_allclose(rho[1, 0], .5*np.exp(-g2.conjugate()), atol=5e-9)
    assert_allclose(np.diag(rho), .5, atol=1e-12)


def test_left_right_pair_reduces_to_classical_noise():
    sigma, gamma, duration = .4, .3, 1.1
    slot = Slot((Segment(.3*X+.1*Z, duration),))
    classical = DenseHEOM([classical_mode(Z/2, sigma, gamma, 12)])
    paired = DenseHEOM(quantum_modes(Z/2, [gamma], [sigma**2], 12))
    rho = np.array([[.7, .2-.1j], [.2+.1j, .3]])
    assert_allclose(root_matrix(paired, rho, slot), root_matrix(classical, rho, slot), atol=1e-9)


def mixed_bath():
    return (*quantum_modes(Z/2, [.5+.2j], [.04+.01j], 3, channel="z"),
            classical_mode(X/2, .2, .1, 3, channel="x"))


def test_mixed_mpo_matches_sum_of_products_and_tdvp_matches_dense():
    modes = mixed_bath()
    whites = [White(Z/2, .03, channel="z")]
    segment = Segment(.3*X+.1*Y, .32, {"z": 1.7, "x": .5})
    tdvp = TDVPHEOM(modes, whites=whites, max_dt=.02, truncation=Truncation(12, 1e-15))
    dense = DenseHEOM(modes, whites=whites)
    nodes = tdvp.generator_mpo(segment)
    # start + end + commutator/left/right of Z/2 + commutator of X/2
    assert nodes[0].shape[-1] == 6
    tensor = nodes[0][0]
    for a in nodes[1:]:
        tensor = np.tensordot(tensor, a, axes=(-1, 0))
    matrix = tensor[..., 0].transpose(0, 2, 4, 6, 1, 3, 5, 7).reshape(108, 108)
    assert_allclose(matrix, dense.generator(segment).toarray(), atol=1e-14)
    slot = Slot((segment, Segment(-.2*Z, .16)))
    for p in (X, Y, Z):
        assert_allclose(tdvp.response(slot, p, tdvp.root()).dense(),
                        dense.response(slot, p, dense.root()).dense(), atol=2e-7)


def test_segment_gain_rescales_coupling_and_zero_switches_it_off():
    modes = [classical_mode(Z/2, .3, .2, 8, channel="z:q0")]
    slot = Slot((Segment(.2*X, .9, {"z:q0": 2.5}),))
    scaled = DenseHEOM(modes)
    reference = DenseHEOM([classical_mode(2.5*Z/2, .3, .2, 8)])
    rho = np.array([[.6, .3j], [-.3j, .4]])
    assert_allclose(root_matrix(scaled, rho, slot),
                    root_matrix(reference, rho, Slot((Segment(.2*X, .9),))), atol=1e-12)
    off = Slot((Segment(.2*X, .9, {"z:q0": 0.}),))
    u = expm(-.9j*.2*X)
    assert_allclose(root_matrix(scaled, rho, off), u@rho@u.conj().T, atol=1e-12)
    # A mode that is off unless a segment names its channel (coupler noise).
    gated = DenseHEOM([classical_mode(Z/2, .3, .2, 8, channel="coupler", idle_gain=0.)])
    assert_allclose(root_matrix(gated, PLUS, idle(1.)), PLUS, atol=1e-12)
    assert abs(root_matrix(gated, PLUS, idle(1., gains={"coupler": 1.}))[0, 1]) < .49


def test_memory_survives_a_segment_with_the_bath_switched_off():
    """Static noise, coupling on-off-on: the off segment must not reset ADOs."""
    sigma, t = .5, .8
    modes = [classical_mode(Z/2, sigma, 0., 12, channel="z")]
    slot = Slot((Segment(np.zeros((2, 2)), t), Segment(np.zeros((2, 2)), 3., {"z": 0.}),
                 Segment(np.zeros((2, 2)), t)))
    rho = root_matrix(DenseHEOM(modes), PLUS, slot)
    assert_allclose(rho[0, 1], .5*np.exp(-.5*sigma**2*(2*t)**2), atol=1e-10)


def test_white_remainder_is_markovian_dephasing():
    rate, duration = .07, 1.3
    engine = DenseHEOM([classical_mode(Z/2, 0, 0, 1)], whites=[White(Z/2, rate, "z")])
    assert_allclose(root_matrix(engine, PLUS, idle(duration))[0, 1],
                    .5*np.exp(-rate*duration), atol=1e-12)
    doubled = root_matrix(engine, PLUS, idle(duration, gains={"z": 2.}))
    assert_allclose(doubled[0, 1], .5*np.exp(-4*rate*duration), atol=1e-12)


def test_mode_validation_and_exponential_table(tmp_path):
    with pytest.raises(ValueError, match="classical"):
        Mode(Z/2, .1+.1j, .2, 3)
    with pytest.raises(ValueError, match="action"):
        Mode(Z/2, .1, .2, 3, "both")
    with pytest.raises(ValueError, match="Hermitian"):
        classical_mode(np.array([[0, 1], [0, 0]]), .1, .2, 3)
    table = tmp_path/"cfdata.dat"
    table.write_text("1 5.5 4.3 1.1e-6 5.8e-7\n2 250.0 0.0 2e-6 0\n3 0.3 -0.01 4e-7 -8e-8\n-1 -2.8e-10\n")
    w, g, k0 = load_exponentials(table, max_decay=100)
    assert_allclose(w, [5.5+4.3j, .3-.01j])
    assert_allclose(g, [1.1e-6+5.8e-7j, 4e-7-8e-8j])
    assert k0 == -2.8e-10
    modes = quantum_modes(Z/2, w, g, 4, channel="z:q0")
    assert [m.action for m in modes] == ["left", "left", "right", "right"]
    assert modes[2].wval == w[0].conjugate() and modes[2].gwval == g[0].conjugate()
    assert len(load_exponentials(table)[0]) == 3


def test_ideal_pulses_reproduce_logical_gates_up_to_phase():
    durations = PulseDurations()
    gates = [Gate("h", (0,)), Gate("x", (0,)), Gate("y", (0,)), Gate("z", (0,)),
             Gate("id", (0,)), Gate("rx", (0,), .37), Gate("ry", (0,), -.8),
             Gate("rz", (0,), 1.1), Gate("cx", (0, 1)), Gate("cx", (1, 0)),
             Gate("cz", (0, 1)), Gate("rzz", (1, 0), .4)]
    for gate in gates:
        h, duration, _ = gate_hamiltonian(gate, 2, durations)
        pulse = expm(-1j*duration*h)
        logical = embed(gate_unitary(gate), gate.qubits, 2) if len(gate.qubits) == 2 \
            else embed(gate_unitary(gate), gate.qubits, 2)
        overlap = np.trace(logical.conj().T@pulse)/4
        assert_allclose(abs(overlap), 1, atol=1e-12, err_msg=gate.name)
    cx = gate_unitary(Gate("cx", (0, 1)))
    assert_allclose(cx@np.eye(4)[2], np.eye(4)[3])  # |10> -> |11>, first qubit controls


def test_ideal_unitary_is_logical_not_pulse_derived():
    """A miscalibrated pulse must show up as an error, not vanish into U_j."""
    rx = Gate("rx", (0,), .5)
    logical = gate_unitary(rx)
    modes = [classical_mode(Z/2, 0., 0., 1)]  # no bath: isolate the coherent error
    over_rotated = Segment(1.1*.5/2*X, 1.)
    explicit = Slot((over_rotated,), "rx", logical)
    fallback = Slot((over_rotated,), "rx")
    assert_allclose(explicit.ideal, logical)
    assert_allclose(fallback.ideal, over_rotated.propagator)
    engine = DenseHEOM(modes)
    fidelity = lambda slot: [engine.response(slot, p, engine.root()).value([0]).real
                             for p in paulis(1)]
    assert_allclose(fidelity(fallback), 1, atol=1e-12)
    # Residual rotation by 0.05 rad about X: Y and Z fidelities are cos(0.05).
    assert_allclose(fidelity(explicit), [1, 1, np.cos(.05), np.cos(.05)], atol=1e-12)
    with pytest.raises(ValueError, match="unitary"):
        Slot((over_rotated,), "bad", 2*logical)


def test_compiled_slots_carry_layer_unitaries_and_channel_gains():
    plan = CircuitLayers(2, (Layer("h-cz", (ParallelBlock((Gate("h", (0,)), Gate("h", (1,)))),
                                            ParallelBlock((Gate("cz", (0, 1)),)))),))
    slots, schedule = compile_layers(plan, PulseDurations(rotation_ns=2, cz_ns=6),
        gains={"cz": {"z:q{1}": 5., "coupler:q{0}q{1}": 1.}})
    assert_allclose(slots[0].ideal, layer_unitary(plan.layers[0], 2))
    assert [s.gains for s in slots[0].segments] == [(), (("coupler:q0q1", 1.), ("z:q1", 5.))]
    assert schedule["layers"][0]["segments"][1]["gains"] == {"z:q1": 5., "coupler:q0q1": 1.}
    assert slots[0].segments[0].key != Segment(slots[0].segments[0].hamiltonian, 2, {"z:q1": 5.}).key


def test_virtual_z_frames_take_no_time_and_match_dense():
    plan = CircuitLayers(1, (Layer("rx-rz-rx", (
        ParallelBlock((Gate("rx", (0,), .5),)), ParallelBlock((Gate("rz", (0,), .9),)),
        ParallelBlock((Gate("rx", (0,), -.3),)))), Layer("z", (ParallelBlock((Gate("z", (0,)),)),))))
    finite, timed = compile_layers(plan, PulseDurations(rotation_ns=.4))
    slots, schedule = compile_layers(plan, PulseDurations(rotation_ns=.4, virtual_z=True))
    assert_allclose([timed["total_duration_ns"], schedule["total_duration_ns"]], [1.6, .8])
    assert [type(s) for s in slots[0].segments] == [Segment, Frame, Segment]
    assert [type(s) for s in slots[1].segments] == [Frame] and slots[1].duration == 0
    assert_allclose(slots[0].ideal, finite[0].ideal)
    modes = [classical_mode(Z/2, .5, .1, 5)]
    dense, tdvp = DenseHEOM(modes), TDVPHEOM(modes, max_dt=.02, truncation=Truncation(5, 1e-15))
    rk4 = TDVPHEOM(modes, integrator="rk4", max_dt=.02, truncation=Truncation(5, 1e-15))
    for slot in slots:
        for p in (X, Y, Z):
            reference = dense.response(slot, p, dense.root()).dense()
            assert_allclose(tdvp.response(slot, p, tdvp.root()).dense(), reference, atol=2e-8)
            assert_allclose(rk4.response(slot, p, rk4.root()).dense(), reference, atol=2e-8)
    # A layer of virtual gates alone is noise-free and keeps the memory intact.
    assert_allclose(dense.response(slots[1], X, dense.root()).dense(), np.eye(5)[0], atol=1e-14)
    p, _ = build_spp(tdvp, slots)
    reference, _ = build_spp(dense, slots)
    assert_allclose(p.dense(), reference.dense(), atol=2e-8)
    # Less exposure to noise than with a finite rz pulse.
    slow, _ = build_spp(DenseHEOM(modes), finite)
    assert p.dense()[0, 0].real > slow.dense()[0, 0].real
