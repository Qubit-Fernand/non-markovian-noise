"""Circuit/noise separation and independent static-hierarchy checks."""
import json
from math import factorial
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm
from heom_spp.circuit import (Gate, ParallelBlock, Layer, CircuitLayers,
    html_ising_layers, from_qiskit, to_qiskit)
from heom_spp.pulse import PulseDurations, compile_layers, system_terms
from heom_spp.bath import Mode, NoiseSpec, build_noise, static_disorder
from heom_spp.heom import DenseHEOM, TDVPHEOM
from heom_spp.pulse import Segment, Slot
from heom_spp.mps import MPS, Truncation
from heom_spp.spp import build_spp
from heom_spp.pauli import X, Z, embed


def test_html_layers_json_and_hamiltonians(tmp_path):
    plan = html_ising_layers(4, steps=2)
    assert len(plan.layers) == 8
    assert [len(x.blocks) for x in plan.layers[:4]] == [2, 2, 2, 1]
    assert plan.layers[2].blocks[-1].gates == (Gate("rz", (2,), -.075),)
    assert plan.layers[3].blocks[0].gates == (Gate("cx", (1, 2)),)
    path = tmp_path/"layers.json"
    plan.save(path)
    assert CircuitLayers.load(path) == plan
    slots, schedule = compile_layers(plan, PulseDurations(rotation_ns=2, cx_ns=4))
    assert_allclose(schedule["total_duration_ns"], 44)
    assert [s.name for s in slots[:4]] == [x.name for x in plan.layers[:4]]
    # Independently build the open-chain Trotter target.
    xx = sum(embed(X, (q,), 4) for q in range(4))
    zz_even = embed(np.kron(Z, Z), (0, 1), 4)+embed(np.kron(Z, Z), (2, 3), 4)
    zz_odd = embed(np.kron(Z, Z), (1, 2), 4)
    step = expm(-1j*(-.075)/2*zz_odd)@expm(-1j*(-.075)/2*zz_even)@expm(-1j*.5/2*xx)
    actual = np.eye(16, dtype=complex)
    for slot in slots:
        actual = slot.ideal@actual
    assert_allclose(actual, step@step, atol=2e-14)


def test_parallel_duration_idle_tail_and_explicit_boundary():
    plan = CircuitLayers(3, (Layer("mixed", (ParallelBlock((
        Gate("rx", (0,), .3), Gate("cx", (1, 2)))),)),))
    slots, schedule = compile_layers(plan, PulseDurations(rotation_ns=1, cx_ns=3))
    assert [s.duration for s in slots[0].segments] == [1, 2]
    assert schedule["layers"][0]["segments"][1]["active_gates"] == ["cx"]
    with pytest.raises(ValueError, match="disjoint"):
        ParallelBlock((Gate("rx", (0,), .3), Gate("cx", (0, 1))))


def test_qiskit_roundtrip_and_bit_order():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator
    plan = html_ising_layers(4)
    qc = to_qiskit(plan)
    recovered = from_qiskit(qc)
    assert recovered == plan
    slots, _ = compile_layers(recovered)
    actual = np.eye(16, dtype=complex)
    for slot in slots:
        actual = slot.ideal@actual
    reversal = [int(f"{i:04b}"[::-1], 2) for i in range(16)]
    reference = Operator(qc).data[np.ix_(reversal, reversal)]
    assert_allclose(actual, reference, atol=1e-14)


def test_qiskit_explicit_gate_counts_and_no_implicit_twirl():
    qiskit = pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    qc = to_qiskit(html_ising_layers(4))
    plain = QuantumCircuit(4)
    for inst in qc.data:
        if inst.operation.name != "barrier":
            plain.append(inst.operation, [qc.find_bit(q).index for q in inst.qubits])
    assert len(from_qiskit(plain).layers) == 1
    explicit = from_qiskit(plain, layer_end_indices=[6, 10, 12, 13])
    assert [x.blocks for x in explicit.layers] == [x.blocks for x in html_ising_layers(4).layers]
    with pytest.raises(ValueError, match="OR"):
        from_qiskit(qc, layer_end_indices=[13])
    measured = QuantumCircuit(1, 1)
    measured.measure(0, 0)
    with pytest.raises(ValueError, match="unitary"):
        from_qiskit(measured)


def test_independent_static_coefficients_and_single_nodes():
    noise = static_disorder(4, sigma=(.1, .2, .3, .4), levels=6)
    assert len(noise.modes) == 4
    assert [m.levels for m in noise.modes] == [6]*4
    assert_allclose([m.gwval for m in noise.modes], [.01, .04, .09, .16])
    assert_allclose([m.wval for m in noise.modes], 0.)
    for q, mode in enumerate(noise.modes):
        assert_allclose(mode.operator, embed(Z, (q,), 4)/2)
    assert noise.summary["stochastic_sampling"] is False
    spec = NoiseSpec.from_dict(dict(schema="heom-noise-v1", n_qubits=4,
                                   gwval=1e-6, wval=0, levels=6))
    assert_allclose([m.sigma for m in build_noise(spec).modes], .001)
    assert spec.to_dict()["wval"] == 0
    assert "sigma" not in spec.to_dict()
    with pytest.raises(ValueError, match="multiple ADO"):
        static_disorder(4, sigma=.1, levels=1)


def test_static_raw_and_scaled_generator_similarity():
    modes = [Mode.from_exponential(Z/2, gwval=.04, wval=0, levels=4)]
    raw = DenseHEOM(modes, hierarchy_basis="unscaled")
    scaled = DenseHEOM(modes)
    segment = Segment(.3*X+.2*Z, .7)
    scale = modes[0].scaling_factors()
    assert_allclose(scale, [.2**n*np.sqrt(factorial(n)) for n in range(4)])
    diagonal = np.tile(scale, 4)
    transformed = raw.generator(segment).toarray()*diagonal[None, :]/diagonal[:, None]
    assert_allclose(transformed, scaled.generator(segment).toarray(), atol=1e-14)


def test_independent_static_ramsey_without_sampling():
    sigma = (.2, .3)
    noise = static_disorder(2, sigma=sigma, levels=10)
    engine = DenseHEOM(noise.modes)
    xx = np.kron(X, X)
    state = MPS([xx.reshape(1, 16, 1, order="F"), *engine.root().cores])
    duration = .8
    final = engine.propagate(state, Slot((Segment(np.zeros((4, 4)), duration),)))
    root = final.dense()[:, 0, 0].reshape(4, 4, order="F")
    measured = np.trace(xx@root)/4
    assert_allclose(measured, np.exp(-.5*sum(x*x for x in sigma)*duration**2), atol=2e-13)


@pytest.mark.parametrize("basis", ["scaled", "unscaled"])
def test_cpu_static_all_ado_response_against_same_basis_dense(basis):
    noise = static_disorder(1, gwval=.09, levels=6, hierarchy_basis=basis)
    cpu = TDVPHEOM(noise.modes, hierarchy_basis=basis, max_dt=.025,
                         truncation=Truncation(6, 1e-15))
    dense = DenseHEOM(noise.modes, hierarchy_basis=basis)
    slot = Slot((Segment(.2*X+.13*Z, .2),))
    for p in (X, Z):
        a = cpu.response(slot, p, cpu.root()).dense()
        b = dense.response(slot, p, dense.root()).dense()
        assert_allclose(a, b, atol=2e-10)


def test_static_spp_raw_scaled_and_joint_gaussian_formula():
    sigma, duration = .4, .7
    slot = Slot((Segment(np.zeros((2, 2)), duration),), "idle")
    result = []
    for basis in ("scaled", "unscaled"):
        noise = static_disorder(1, sigma=sigma, levels=10, hierarchy_basis=basis)
        engine = DenseHEOM(noise.modes, hierarchy_basis=basis)
        p, _ = build_spp(engine, [slot, slot],
            time_truncation=Truncation(10, 1e-14),
            assembly_truncation=Truncation(10, 1e-14))
        result.append(p.dense())
    assert_allclose(result[0], result[1], atol=1e-13)
    c = np.exp(-.5*sigma**2*duration**2)
    c2 = np.exp(-2*sigma**2*duration**2)
    pz = (1-c)/2
    pzz = (3-4*c+c2)/8
    expected = np.zeros((4, 4))
    expected[3, 3] = pzz
    expected[3, 0] = expected[0, 3] = pz-pzz
    expected[0, 0] = 1-2*pz+pzz
    assert_allclose(result[0], expected, atol=1e-12)
    assert pzz-pz*pz > 1e-4  # static memory survives ideal layer twirling
