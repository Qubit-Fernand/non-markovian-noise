"""sqisw decomposition, square-pulse compilation, frames, and kernel equivalence."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm
from heom_spp.bath import classical_mode, quantum_modes
from heom_spp.circuit import (SQISW, CircuitLayers, Gate, gate_unitary,
                              layer_unitary, from_qiskit, to_qiskit)
from heom_spp.heom import DenseHEOM, TDVPHEOM
from heom_spp.mps import Truncation
from heom_spp.native import (equal_up_to_phase, ising_sqisw_layers,
                             ising_trotter_unitary, pulse_gates, zxz, zz_locals)
from heom_spp.pauli import X, Y, Z, embed
from heom_spp.pulse import (SQUARE_NATIVE, Frame, PulseDurations,
                            RotatingFrames, Segment, compile_layers)
from heom_spp import tdvp


def total_unitary(plan):
    u = np.eye(2**plan.n_qubits, dtype=complex)
    for layer in plan.layers:
        u = layer_unitary(layer, plan.n_qubits)@u
    return u


def pulse_unitary(slot):
    u = np.eye(slot.segments[0].dim, dtype=complex)
    for operation in slot.segments:
        u = operation.propagator@u
    return u


def test_single_qubit_unitary_is_one_xy_pulse_and_an_ideal_z():
    rng = np.random.default_rng(1)
    cases = [expm(-1j*sum(c*p for c, p in zip(rng.normal(size=3), (X, Y, Z)))) for _ in range(20)]
    cases += [np.eye(2), X, Y, Z, expm(-.3j*Z), (X+Z)/np.sqrt(2)]
    for u in cases:
        a, t, b = zxz(u)
        assert 0 <= t <= np.pi+1e-12
        pulse, z = pulse_gates(u, 0)
        assert len(pulse) <= 1 and len(z) <= 1
        built = np.eye(2, dtype=complex)
        for g in pulse+z:
            built = gate_unitary(g)@built
        assert equal_up_to_phase(built, u)
    assert pulse_gates(np.eye(2), 0) == ([], [])
    assert pulse_gates(expm(-.3j*Z), 0)[0] == []  # pure Z needs no pulse


@pytest.mark.parametrize("theta", [-.075, .3, -1.1, np.pi/2])
def test_two_sqisw_reach_any_small_zz_rotation(theta):
    k1, k2, m1, m2, k3, k4 = zz_locals(theta)
    built = np.kron(k3, k4)@SQISW@np.kron(m1, m2)@SQISW@np.kron(k1, k2)
    assert equal_up_to_phase(built, expm(-.5j*theta*np.kron(Z, Z)), 1e-11)
    # The interleaved pair is the closed form quoted in the module docstring.
    alpha = np.pi-2*np.arcsin(min(1., np.sqrt(2)*np.sin(abs(theta)/2)))
    assert equal_up_to_phase(m1, expm(-.5j*alpha*X)) and equal_up_to_phase(m2, Y)
    with pytest.raises(ValueError, match="pi/2"):
        zz_locals(2.)


def test_sqisw_convention():
    assert_allclose(SQISW@SQISW@SQISW@SQISW, np.diag([1, -1, -1, 1]), atol=1e-12)
    assert_allclose(SQISW@np.eye(4)[1], (np.eye(4)[1]-1j*np.eye(4)[2])/np.sqrt(2))


@pytest.mark.parametrize("n,steps", [(2, 1), (3, 1), (4, 1), (4, 2)])
def test_native_trotter_step_equals_the_logical_one(n, steps):
    plan = ising_sqisw_layers(n, theta_x=.5, theta_z=-.075, steps=steps)
    assert equal_up_to_phase(total_unitary(plan), ising_trotter_unitary(n, .5, -.075, steps), 1e-10)
    names = {g.name for layer in plan.layers for block in layer.blocks for g in block.gates}
    assert names <= {"r", "rz", "sqisw"}
    assert len(plan.layers) == (2 if n == 2 else 4)*steps
    counts = [sum(g.name == "sqisw" for b in layer.blocks for g in b.gates) for layer in plan.layers]
    assert counts[:4] == {2: [1, 1], 3: [1, 1, 1, 1], 4: [2, 2, 1, 1]}[n][:4]


def test_square_pulse_schedule_timing_and_amplitudes(tmp_path):
    plan = ising_sqisw_layers(4)
    slots, schedule = compile_layers(plan, SQUARE_NATIVE)
    # 5 pulse rounds of 15 ns, 4 exchange rounds of 20 ns, 9 buffers of 5 ns.
    assert_allclose(schedule["total_duration_ns"], 5*15+4*20+9*5)
    assert_allclose([s.duration for s in slots], [45, 45, 45, 65])
    for slot in slots:
        assert equal_up_to_phase(pulse_unitary(slot), slot.ideal, 1e-11)
        assert any(isinstance(s, Frame) for s in slot.segments)
    for p in schedule["pulses"]:
        duration = p["end_ns"]-p["start_ns"]
        if p["gate"] == "sqisw":
            assert_allclose([duration, p["g"]*duration], [20, np.pi/4])
        else:
            assert_allclose(duration, 15)
            assert_allclose(np.hypot(p["omega_x"], p["omega_y"])*duration, p["angle"])
    assert all(z["time_ns"] in {p["end_ns"] for p in schedule["pulses"]} for z in schedule["ideal_z"])
    path = tmp_path/"native.json"
    plan.save(path)
    assert CircuitLayers.load(path) == plan


def test_qiskit_roundtrip_of_native_gates():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator
    plan = ising_sqisw_layers(4)
    qc = to_qiskit(plan)
    assert from_qiskit(qc) == plan
    reversal = [int(f"{i:04b}"[::-1], 2) for i in range(16)]
    assert equal_up_to_phase(Operator(qc).data[np.ix_(reversal, reversal)], total_unitary(plan), 1e-10)


def test_frame_changes_are_listed_and_only_matter_when_uncompensated():
    plan = ising_sqisw_layers(2)
    plain, _ = compile_layers(plan, SQUARE_NATIVE)
    frames = RotatingFrames((1,), detuning=2*np.pi*.4)
    compensated, schedule = compile_layers(plan, SQUARE_NATIVE, frames=frames)
    changes = schedule["frame_changes"]
    assert [c["change"] for c in changes] == ["enter", "leave"]*2
    assert all(c["qubit"] == 1 and c["partner"] == 0 for c in changes)
    assert_allclose([c["frame_angle"] for c in changes],
                    [s*frames.detuning*c["time_ns"] for s, c in zip((-1, 1, -1, 1), changes)])
    assert_allclose([c["frame_angle"]+c["compensation_angle"] for c in changes], 0)
    for a, b in zip(plain, compensated):
        assert_allclose(pulse_unitary(a), pulse_unitary(b))
    raw, _ = compile_layers(plan, SQUARE_NATIVE, frames=RotatingFrames((1,), .37, compensate=False))
    assert not equal_up_to_phase(pulse_unitary(raw[0]), raw[0].ideal, 1e-3)
    # The uncorrected gate is the exchange gate conjugated by the frame phases.
    t0, t1 = changes[0]["time_ns"], changes[1]["time_ns"]
    rz = lambda angle: embed(expm(-.5j*angle*Z), (1,), 2)
    blocks = plan.layers[0].blocks
    before = np.eye(4, dtype=complex)
    for block in blocks[:-1]:
        for g in block.gates:
            before = embed(gate_unitary(g), g.qubits, 2)@before
    assert equal_up_to_phase(pulse_unitary(raw[0]), rz(.37*t1)@SQISW@rz(-.37*t0)@before, 1e-11)
    with pytest.raises(ValueError, match="tunable"):
        compile_layers(plan, SQUARE_NATIVE, frames=RotatingFrames((0, 1)))


def test_gemm_kernels_equal_the_reference_einsums():
    rng = np.random.default_rng(5)
    c = lambda *shape: rng.normal(size=shape)+1j*rng.normal(size=shape)
    a, b, d, e, w, z, i, o = 3, 4, 5, 2, 3, 2, 6, 6
    left, right, op, x = c(a, w, b), c(d, z, e), c(w, o, i, z), c(b, i, e)
    packed = tdvp.pack(op)
    assert_allclose(tdvp._apply_k(x, left, packed, tdvp._pack_right_k(right)),
                    np.einsum("awb,woiz,bic,dzc->aod", left, op, x, right), atol=1e-12)
    bond = c(b, e)
    env = c(d, w, e)
    assert_allclose(tdvp._apply_s(bond, left, tdvp._pack_right_s(env)),
                    -np.einsum("awb,bc,dwc->ad", left, bond, env), atol=1e-12)
    core = c(a, i, 5)
    assert_allclose(tdvp._left(c(a, w, a), packed, core).shape, (5, z, 5))
    environment = c(a, w, a)
    assert_allclose(tdvp._left(environment, packed, core),
                    np.einsum("awb,aod,woiz,bic->dzc", environment, core.conj(), op, core), atol=1e-11)
    core, environment = c(4, i, 3), c(3, z, 3)
    assert_allclose(tdvp._right(environment, packed, core),
                    np.einsum("aod,woiz,bic,dzc->awb", core.conj(), op, core, environment), atol=1e-11)


def test_trimmed_mpo_is_the_same_operator_and_is_narrower_for_static_modes():
    def dense(nodes):
        tensor = nodes[0][0]
        for a in nodes[1:]:
            tensor = np.tensordot(tensor, a, axes=(-1, 0))
        return tensor[..., 0]
    segment = Segment(.2*embed(X, (0,), 2)+.1*embed(Z, (1,), 2), .5)
    cases = {
        "static": [classical_mode(embed(Z, (q,), 2)/2, .1, 0., 3) for q in range(2)],
        "mixed": [*quantum_modes(embed(Z, (0,), 2)/2, [.5+.2j], [.04+.01j], 2),
                  classical_mode(embed(Z, (1,), 2)/2, .1, 0., 3)],
    }
    for name, modes in cases.items():
        trimmed, full = TDVPHEOM(modes), TDVPHEOM(modes)
        full.trim = False
        assert_allclose(dense(trimmed.generator_mpo(segment)), dense(full.generator_mpo(segment)),
                        atol=1e-14, err_msg=name)
    static = TDVPHEOM(cases["static"])
    # start-identity is never needed: [comm Z0, comm Z1, end] then [comm Z1, end].
    assert [w.shape[-1] for w in static.generator_mpo(segment)] == [3, 2, 1]
    engine, reference = TDVPHEOM(cases["mixed"], max_dt=.02, truncation=Truncation(8, 1e-15)), \
        DenseHEOM(cases["mixed"])
    from heom_spp.pulse import Slot
    slot = Slot((segment,))
    for p in (embed(X, (0,), 2), np.kron(Y, Z)):
        assert_allclose(engine.response(slot, p, engine.root()).dense(),
                        reference.response(slot, p, reference.root()).dense(), atol=2e-7)
