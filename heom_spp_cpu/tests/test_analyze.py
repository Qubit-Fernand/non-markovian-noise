"""Mitigation analysis on a saved SPP: exact expectations, ZNE, PEC, Gamma."""
import json
import numpy as np
import pytest
from numpy.testing import assert_allclose
from heom_spp.analyze import (analyze, expectation, gamma_estimate, pauli_label,
                              pauli_transfer, table, values_at)
from heom_spp.bath import classical_mode
from heom_spp.circuit import CircuitLayers, Gate, Layer, ParallelBlock, layer_unitary
from heom_spp.heom import DenseHEOM
from heom_spp.mitigation import inverse_dense, response_tensor
from heom_spp.mps import MPS
from heom_spp.pauli import X, Z, paulis, walsh
from heom_spp.pulse import PulseDurations, compile_layers
from heom_spp.spp import build_spp, independent_model


@pytest.fixture(scope="module")
def saved_run(tmp_path_factory):
    """One qubit, three layers, strong static noise: clearly correlated SPP."""
    block = lambda name, angle: ParallelBlock((Gate(name, (0,), angle),))
    plan = CircuitLayers(1, (Layer("a", (block("rx", .7),)), Layer("b", (block("ry", -.4),)),
                             Layer("c", (block("rx", .9),))))
    slots, _ = compile_layers(plan, PulseDurations(rotation_ns=1.))
    p, _ = build_spp(DenseHEOM([classical_mode(Z/2, .5, 0., 10)]), slots)
    run = tmp_path_factory.mktemp("run")
    p.save(run/"spp.npz")
    plan.save(run/"layers.json")
    return run, plan, slots, p


def test_exact_expectation_matches_history_enumeration(saved_run):
    run, plan, slots, p = saved_run
    basis = paulis(1)
    transfers = [pauli_transfer(layer_unitary(layer, 1), basis) for layer in plan.layers]
    for t in transfers:
        assert_allclose(t@t.T, np.eye(4), atol=1e-12)
    psi = np.array([1, 0], complex)
    rho = np.einsum("i,bij,j->b", psi.conj(), basis, psi).real/2
    for word, operator in (("X", X), ("Z", Z)):
        observable = np.zeros(4)
        observable[pauli_label(word)] = 2
        response = response_tensor(slots, operator, psi)
        assert_allclose(expectation(walsh(p, 1), transfers, rho, observable),
                        np.sum(p.dense()*response).real, atol=1e-12)
        assert_allclose(expectation(MPS.product([np.ones(4)]*3), transfers, rho, observable),
                        response[0, 0, 0], atol=1e-12)
    assert pauli_label("IZ") == 3 and pauli_label("XI") == 4
    with pytest.raises(ValueError):
        pauli_label("ZQ")


def test_analysis_pipeline_zne_and_pec(saved_run):
    run, plan, slots, p = saved_run
    result = analyze(run, observables=["X", "Z"], gains=(1, 2, 3), pec_bonds=(1, 4),
                     shots=4000, progress=lambda *_: None)
    json.dumps(result)  # everything is serialisable
    assert "PEC SPP bond 4" in table(result)
    ideal, noisy = result["ideal"], result["noisy"]
    error = lambda values: max(abs(values[w]-ideal[w]) for w in ideal)
    assert error(noisy) > 1e-2
    assert result["correlation_l2"] > 1e-3
    # Full-bond PEC removes the noise; the layer-wise inverse cannot.
    full, product = result["pec"]["correlated"][1], result["pec"]["independent"]
    assert error(full["mitigated"]) < 1e-6 < error(product["mitigated"])
    assert full["residual_rms"] < 1e-6 < product["residual_rms"]
    # Gamma against exhaustive enumeration of the exact inverse.
    exact = np.abs(inverse_dense(p, 1).dense().real).sum()
    assert abs(full["gamma"]-exact) < 5*full["gamma_standard_error"]+1e-3
    assert_allclose(product["gamma"], np.prod(product["gamma_per_layer"]))
    # Amplifying with the SPP extrapolates better than with the independent model.
    zne = result["zne"]
    assert_allclose([zne["correlated"][0][w] for w in ideal], [noisy[w] for w in ideal])
    assert error(zne["extrapolated"]["correlated"]["order2"]) < error(noisy)
    assert error(zne["extrapolated"]["correlated"]["order2"]) \
        < error(zne["extrapolated"]["independent"]["order2"])


def test_batched_values_and_gamma_estimator():
    rng = np.random.default_rng(3)
    q = MPS.from_dense(rng.normal(size=(4, 4, 4)))
    histories = rng.integers(4, size=(50, 3))
    assert_allclose(values_at(q, histories), [q.value([int(x) for x in h]) for h in histories], atol=1e-12)
    proposal = [np.full(4, .25)]*3
    gamma, error = gamma_estimate(q, proposal, 40000, rng)
    assert abs(gamma-np.abs(q.dense().real).sum()) < 5*error
    product = independent_model(MPS.product([np.array([.7, .1, .1, .1])]*2))
    assert_allclose(gamma_estimate(product, [np.array([.7, .1, .1, .1])]*2, 100, rng), (1, 0), atol=1e-12)


def test_factors_exact_zne_progress_and_step_plot(saved_run, tmp_path):
    import shutil
    import sys
    from pathlib import Path
    run, plan, slots, p = saved_run
    basis = paulis(1)
    transfers = [pauli_transfer(layer_unitary(layer, 1), basis) for layer in plan.layers]
    rho, observable = np.array([.5, 0, 0, .5]), np.array([0, 2., 0, 0])
    f = walsh(p, 1)
    # Factors contracted side by side == their entrywise product formed first.
    assert_allclose(expectation([f, f, f], transfers, rho, observable),
                    expectation(f.hadamard(f).hadamard(f), transfers, rho, observable), atol=1e-13)
    lines = []
    exact = analyze(run, observables=["X", "Z"], pec_bonds=(1, 2), sweeps=3, shots=200,
                    progress=lines.append)
    forced = analyze(run, observables=["X", "Z"], pec_bonds=(1,), sweeps=1, shots=200,
                     progress=lambda *_: None, exact_entries=0)  # compressed path
    assert exact["zne"]["truncation_discarded"] == [0., 0., 0.] and exact["seconds"] > 0
    for a, b in zip(exact["zne"]["correlated"], forced["zne"]["correlated"]):
        assert_allclose([a[w] for w in a], [b[w] for w in a], atol=1e-10)
    assert any("PEC bond 1: sweep 1/3" in x and "left]" in x for x in lines)
    assert all(x.startswith("[0:00:0") for x in lines)
    # Step scan layout -> table, csv and (with matplotlib) the figure.
    sys.path.insert(0, str(Path(__file__).parents[1]/"scripts"))
    import plot_steps
    for steps in (1, 2):
        (tmp_path/f"steps{steps}").mkdir()
        (tmp_path/f"steps{steps}"/"mitigation.json").write_text(json.dumps(exact), encoding="utf-8")
    plot_steps.main([str(tmp_path)])
    rows = plot_steps.collect(tmp_path, "max", None, "order2")
    assert [r["steps"] for r in rows] == [1, 2] and rows[0]["PEC SPP bond 2"] < rows[0]["PEC SPL"]
    assert (tmp_path/"steps.csv").exists()
    if shutil.which("python") and "matplotlib" in sys.modules:
        assert (tmp_path/"steps.png").stat().st_size > 10_000


def test_unnormalised_spp_is_normalised_and_reported(saved_run, tmp_path):
    run, plan, slots, p = saved_run
    p.scaled(.98).save(tmp_path/"spp.npz")
    plan.save(tmp_path/"layers.json")
    lines = []
    kwargs = dict(observables=["X", "Z"], pec_bonds=(), shots=100)
    a = analyze(run, progress=lambda *_: None, **kwargs)
    b = analyze(tmp_path, progress=lines.append, **kwargs)
    assert abs(b["total_probability"]-.98) < 1e-12 and any("WARNING" in x for x in lines)
    for key in ("noisy", "noisy_independent_model"):
        assert_allclose([b[key][w] for w in "XZ"], [a[key][w] for w in "XZ"], atol=1e-12)
    assert_allclose(b["pec"]["independent"]["gamma"], a["pec"]["independent"]["gamma"])
    assert_allclose([b["pec"]["independent"]["mitigated"][w] for w in "XZ"],
                    [a["pec"]["independent"]["mitigated"][w] for w in "XZ"], atol=1e-12)


@pytest.fixture
def baseline_pair(tmp_path):
    plan = CircuitLayers(2, (
        Layer("drive", (ParallelBlock((Gate("r", (0,), .7, .2), Gate("rx", (1,), .3))),)),
        Layer("entangle", (ParallelBlock((Gate("cx", (0, 1)),)),))))
    paths = [tmp_path/"device", tmp_path/"baseline"]
    for i, path in enumerate(paths):
        path.mkdir()
        plan.save(path/"layers.json")
        # Different noise models on the same circuit must remain valid baselines.
        weights = np.full(16, (.1+.02*i)/15)
        weights[0] = .9-.02*i
        MPS.product([weights, weights]).save(path/"spp.npz")
        (path/"noise.json").write_text(json.dumps({"sigma": .1+.1*i}))
    return paths


@pytest.mark.parametrize("change", ["gate", "angle", "phase", "qubits", "layer_order", "blocks", "boundary"])
def test_baseline_rejects_same_shape_different_circuit(baseline_pair, change):
    run, baseline = baseline_pair
    data = json.loads((baseline/"layers.json").read_text())
    gate = data["layers"][0]["blocks"][0]["gates"][0]
    if change == "gate":
        data["layers"][1]["blocks"][0]["gates"][0]["name"] = "cz"
    elif change in ("angle", "phase"):
        gate[change] += .01
    elif change == "qubits":
        data["layers"][1]["blocks"][0]["gates"][0]["qubits"] = [1, 0]
    elif change == "layer_order":
        data["layers"].reverse()
    elif change == "boundary":
        moved = data["layers"][0]["blocks"][0]["gates"].pop()
        data["layers"][1]["blocks"].insert(0, {"gates": [moved]})
    else:
        # Same ideal unitary, different parallel/sequential block definition.
        gates = data["layers"][0]["blocks"][0]["gates"]
        data["layers"][0]["blocks"] = [{"gates": [g]} for g in gates]
    (baseline/"layers.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="baseline comparison: layers.json circuit/layer"):
        analyze(run, observables=["ZI"], baselines={"comparison": baseline},
                pec_bonds=(), shots=100, progress=lambda *_: None)
    assert not (run/"mitigation.json").exists()


def test_baseline_accepts_same_circuit_with_new_metadata_and_noise(baseline_pair):
    run, baseline = baseline_pair
    data = json.loads((baseline/"layers.json").read_text())
    data["source"] = "another export"
    for layer in data["layers"]:
        layer["name"] = "renamed"
        layer["step"] += 10
        for block in layer["blocks"]:
            block["gates"].reverse()  # disjoint parallel gates commute
    (baseline/"layers.json").write_text(json.dumps(data, indent=4))
    result = analyze(run, observables=["ZI"], baselines={"comparison": baseline},
                     gains=(1, 2), pec_bonds=(), shots=100, progress=lambda *_: None)
    assert "comparison" in result["pec"]
    assert "comparison" in result["zne"]["extrapolated"]
    assert result["baseline_distance_to_marginals_l2"]["comparison"] > 0


@pytest.mark.parametrize("damage", ["missing", "invalid", "wrong_order"])
def test_baseline_requires_valid_layer_metadata(baseline_pair, damage):
    run, baseline = baseline_pair
    path = baseline/"layers.json"
    if damage == "missing":
        path.rename(baseline/"original-layers.json")
    elif damage == "invalid":
        path.write_text("{")
    else:
        data = json.loads(path.read_text())
        data["qubit_order"] = "q0-least-significant"
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="baseline comparison: cannot validate"):
        analyze(run, observables=["ZI"], baselines={"comparison": baseline},
                pec_bonds=(), shots=100, progress=lambda *_: None)
