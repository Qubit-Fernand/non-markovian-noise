"""Independent correctness checks for CPU TDVP, gate-after, and multiprocessing."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm
from heom_spp.tdvp import arnoldi_action
from heom_spp.bath import classical_mode
from heom_spp.heom import DenseHEOM, HEOM, TDVPHEOM
from heom_spp.pulse import Segment, Slot
from heom_spp.spp import build_spp
from heom_spp.mps import MPS, Truncation
from heom_spp.pauli import I, X, Y, Z


def test_nonhermitian_arnoldi_zero_and_breakdown():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(7, 7))+1j*rng.normal(size=(7, 7))
    v = rng.normal(size=7)+1j*rng.normal(size=7)
    result, _ = arnoldi_action(lambda x: a@x, v, .2, dimension=7, tolerance=1e-12)
    assert_allclose(result, expm(.2*a)@v, atol=1e-12)
    assert_allclose(arnoldi_action(lambda x: a@x, np.zeros(7), .2)[0], 0)
    assert_allclose(arnoldi_action(lambda x: 2*x, v, .2)[0], np.exp(.4)*v)


def model(dt=.04):
    modes = [classical_mode(Z/2, .31, .2, 3), classical_mode(Z/2, .17, .07, 3)]
    jump = np.array([[0, 1], [0, 0]], complex)*np.sqrt(.12)
    engine = TDVPHEOM(modes, max_dt=dt, truncation=Truncation(9, 1e-14),
                           jumps=[jump])
    slot = Slot((Segment(.3*X+.1*Y, .24), Segment(-.13*Z, .16)), "non-Clifford")
    dense = DenseHEOM(modes, jumps=[jump])
    return engine, dense, slot


def test_grouped_mpo_matches_independent_generator():
    engine, dense, slot = model()
    nodes = engine.generator_mpo(slot.segments[0])
    assert nodes[0].shape[-1] == 3  # two modes, one Q, start + Q + end
    tensor = nodes[0][0]
    for a in nodes[1:]:
        tensor = np.tensordot(tensor, a, axes=(-1, 0))
    tensor = tensor[..., 0]
    # axes s_out,s_in,k1_out,k1_in,k2_out,k2_in
    matrix = tensor.transpose(0, 2, 4, 1, 3, 5).reshape(36, 36)
    assert_allclose(matrix, dense.generator(slot.segments[0]).toarray(), atol=1e-14)


def test_tdvp_all_ados_gate_after_and_time_refinement():
    errors = []
    for dt in (.08, .04):
        engine, dense, slot = model(dt)
        u = slot.ideal
        probes = [I, X, Y, Z]
        maximum = 0.
        for p in probes:
            actual = engine.response(slot, p, engine.root()).dense()
            reference = dense.response(slot, p, dense.root()).dense()
            maximum = max(maximum, np.max(abs(actual-reference)))
        errors.append(maximum)
    assert errors[1] < errors[0]/3
    assert errors[1] < 2e-7


def test_sequential_and_tree_spp_match_upstream_and_dense():
    engine, dense, slot = model(.04)
    slots = [slot, slot]
    cutoff = Truncation(9, 1e-13)
    reference, _ = build_spp(dense, slots, time_truncation=cutoff,
                            assembly_truncation=cutoff)
    original, _ = build_spp(HEOM(engine.modes, max_dt=.04, jumps=[
        np.array([[0, 1], [0, 0]], complex)*np.sqrt(.12)]),
        slots, time_truncation=cutoff, assembly_truncation=cutoff)
    assert_allclose(original.dense(), reference.dense(), atol=2e-7)
    for method in ("sequential", "tree"):
        e, _, _ = model(.04)
        p, _ = build_spp(e, slots, assembly=method, chunk_size=2,
                            time_truncation=cutoff, assembly_truncation=cutoff)
        assert_allclose(p.dense(), reference.dense(), atol=2e-7)
        assert_allclose(p.total(), 1., atol=1e-12)


def test_spawn_parallel_shared_memory_matches_serial():
    engine, _, slot = model(.08)
    options = dict(time_truncation=Truncation(9, 1e-13),
                   assembly_truncation=Truncation(9, 1e-13), chunk_size=2)
    serial, _ = build_spp(engine, [slot, slot], workers=1, **options)
    parallel_engine, _, _ = model(.08)
    parallel, records = build_spp(parallel_engine, [slot, slot], workers=2, **options)
    assert_allclose(parallel.dense(), serial.dense(), atol=1e-12)
    assert parallel_engine.stats.responses == engine.stats.responses
    assert [r["probes"] for r in records] == [4, serial.cores[0].shape[-1]*4]


def test_parallel_failure_propagates_without_unlinking_pending_work_early():
    engine, _, slot = model(.08)
    engine.krylov_dim = 1
    with pytest.raises(RuntimeError, match="Arnoldi budget"):
        build_spp(engine, [slot], workers=2, chunk_size=1)
