"""Quadrature reference for static noise, bath resets, and their use as baselines."""
import json
import numpy as np
import pytest
from numpy.testing import assert_allclose
from heom_spp import cli
from heom_spp.analyze import analyze
from heom_spp.bath import classical_mode
from heom_spp.circuit import CircuitLayers, Gate, Layer, ParallelBlock
from heom_spp.heom import HEOM, DenseHEOM, TDVPHEOM
from heom_spp.mps import MPS, Truncation
from heom_spp.pauli import X, Z, embed
from heom_spp.pulse import PulseDurations, Reset, Segment, compile_layers
from heom_spp.reference import mixture_mps, static_nodes, static_reference, static_reference_check
from heom_spp.spp import build_spp, independent_model


def plan(n=1):
    gate = lambda name, angle, *qubits: Gate(name, qubits, angle)
    blocks = lambda *gates: tuple(ParallelBlock((g,)) for g in gates)
    if n == 1:
        return CircuitLayers(1, (Layer("a", blocks(gate("rx", .7, 0), gate("ry", .3, 0))),
                                 Layer("b", blocks(gate("ry", -.4, 0))),
                                 Layer("c", blocks(gate("rx", .9, 0), gate("rx", .2, 0)))))
    return CircuitLayers(2, (Layer("a", blocks(gate("rx", .7, 0), gate("ry", .3, 1))),
                             Layer("b", blocks(Gate("sqisw", (0, 1)), gate("rx", .5, 1)))))


def modes(n, sigma=.3, levels=14):
    return [classical_mode(embed(Z/2, (q,), n), sigma*(1+.3*q), 0., levels) for q in range(n)]


@pytest.mark.parametrize("n", [1, 2])
def test_quadrature_matches_heom(n):
    slots, _ = compile_layers(plan(n), PulseDurations(rotation_ns=1., sqisw_ns=1.5))
    engine = DenseHEOM(modes(n), max_entries=10**6)
    p, _ = build_spp(engine, slots)
    q, info = static_reference_check(engine, slots, nodes=16)
    assert info["quadrature_check_l2"] < 1e-9 and info["unitarity_error"] < 1e-12
    assert abs(q.total()-1) < 1e-12
    assert p.add(q.scaled(-1)).norm() < 1e-7
    assert q.add(independent_model(q).scaled(-1)).norm() > 1e-3  # clearly correlated


def test_mixture_compression_and_guards():
    rng = np.random.default_rng(1)
    tables = [rng.random((30, 4)) for _ in range(4)]
    weights = rng.random(30)
    exact = np.einsum("n,na,nb,nc,nd->abcd", weights, *tables)
    p, discarded = mixture_mps(tables, weights)
    assert_allclose(p.dense(), exact, atol=1e-12)
    assert discarded < 1e-12 and p.bonds == (4, 16, 4)
    delta, w = static_nodes(modes(2), 5)
    assert delta.shape == (25, 2) and abs(w.sum()-1) < 1e-14
    assert_allclose(w@delta**2, [.3**2, .39**2], atol=1e-14)
    slots, _ = compile_layers(plan(), PulseDurations(rotation_ns=1.))
    with pytest.raises(ValueError):
        static_nodes([classical_mode(Z/2, .1, .05, 4)], 4)       # not static
    with pytest.raises(ValueError):
        static_reference(DenseHEOM(modes(1), jumps=[.1*X]), slots)  # not closed
    resets, _ = compile_layers(plan(), PulseDurations(rotation_ns=1.), reset_blocks=True)
    with pytest.raises(ValueError):
        static_reference(DenseHEOM(modes(1)), resets)


def test_layer_reset_is_the_product_of_marginals_for_classical_noise():
    slots, _ = compile_layers(plan(2), PulseDurations(rotation_ns=1., sqisw_ns=1.5))
    engine = DenseHEOM(modes(2, levels=8), max_entries=10**6)
    p, _ = build_spp(engine, slots)
    product, records = build_spp(engine, slots, reset=True)
    assert product.bonds == (1,) and [r["probes"] for r in records] == [16, 16]
    assert product.add(independent_model(p).scaled(-1)).norm() < 1e-10


def test_pulse_reset_cuts_coherent_accumulation_inside_a_layer():
    # Two idle blocks of time t in one layer: Z error probability
    # (1 - exp(-s^2 T^2 / 2)) / 2 with T = 2t together, but two independent
    # blocks of t each when the bath is reset in between.
    sigma, t = .2, 1.
    layer = Layer("idle", (ParallelBlock((Gate("id", (0,)),)), ParallelBlock((Gate("id", (0,)),))))
    layers = CircuitLayers(1, (layer,))
    flip = lambda time: (1-np.exp(-sigma**2*time**2/2))/2
    durations = PulseDurations(idle_ns=t)
    bath = [classical_mode(Z/2, sigma, 0., 14)]
    whole, _ = compile_layers(layers, durations)
    cut, schedule = compile_layers(layers, durations, reset_blocks=True)
    assert [type(s) for s in cut[0].segments] == [Segment, Reset, Segment, Reset]
    assert sum("bath_reset" in e for e in schedule["layers"][0]["segments"]) == 2
    separate = 2*flip(t)*(1-flip(t))
    assert abs(separate-flip(2*t)) > 1e-3
    for engine in (DenseHEOM(bath), HEOM(bath, max_dt=.05),
                   TDVPHEOM(bath, max_dt=.05, truncation=Truncation(4, 1e-13))):
        tolerance = 1e-9 if isinstance(engine, DenseHEOM) else 1e-6
        assert_allclose(build_spp(engine, whole)[0].dense()[3].real, flip(2*t), atol=tolerance)
        assert_allclose(build_spp(engine, cut, reset=True)[0].dense()[3].real, separate, atol=tolerance)


def test_cli_reference_reset_converge_and_baselines(tmp_path, capsys):
    common = ["--example", "one-qubit", "--steps", "2", "--sigma", ".02", "--rotation-ns", "10"]
    heom = ["--engine", "dense", "--levels", "8"]
    cli.main(["reference", *common, "--nodes", "14", "--output", str(tmp_path/"reference")])
    cli.main(["run", *common, *heom, "--output", str(tmp_path/"base")])
    cli.main(["run", *common, "--engine", "dense", "--levels", "2", "--output", str(tmp_path/"lv2")])
    cli.main(["run", *common, *heom, "--reset", "layer", "--output", str(tmp_path/"reset-layer")])
    cli.main(["run", *common, *heom, "--reset", "pulse", "--output", str(tmp_path/"reset-pulse")])
    summary = json.loads((tmp_path/"reference"/"summary.json").read_text(encoding="utf-8"))
    assert summary["kind"] == "reference" and summary["quadrature"]["quadrature_check_l2"] < 1e-10
    capsys.readouterr()
    cli.main(["converge", str(tmp_path)])
    out = capsys.readouterr().out
    report = json.loads((tmp_path/"convergence.json").read_text(encoding="utf-8"))
    assert "recommended (fastest run within tolerance): base" in out
    assert sorted(report["skipped"]) == ["reset-layer", "reset-pulse"]
    errors = {r["name"]: r["joint_l2"] for r in report["runs"]}
    assert errors["base"] < report["tolerance"] < errors["lv2"]
    assert report["matching_run"].endswith("base")
    # Baselines enter the mitigation analysis as extra independent models.
    result = analyze(tmp_path/"base", observables=["Z"], pec_bonds=(1,), shots=200,
                     progress=lambda *_: None, baselines={
                         "reset-layer": tmp_path/"reset-layer", "reset-pulse": tmp_path/"reset-pulse"})
    assert result["baseline_distance_to_marginals_l2"]["reset-layer"] < 1e-10
    assert_allclose(result["pec"]["reset-layer"]["mitigated"]["Z"],
                    result["pec"]["independent"]["mitigated"]["Z"], atol=1e-10)
    assert set(result["zne"]["extrapolated"]) == {"correlated", "independent", "reset-layer", "reset-pulse"}
    assert set(MPS.load(tmp_path/"reset-pulse"/"spp.npz").bonds) == {1}
