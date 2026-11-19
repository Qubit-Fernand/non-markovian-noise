"""Regression tests for stale convergence caches, including the real shell entry."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from heom_spp import cli
from heom_spp.bath import NoiseSpec
from heom_spp.reuse import check_cached, check_scan


COMMON = ["--example", "one-qubit", "--rotation-ns", "1", "--steps", "1",
          "--bath", "static", "--sigma", ".01"]
NUMERICAL = ["--levels", "2", "--engine", "tdvp", "--max-dt", ".1",
             "--heom-bond", "4", "--time-bond", "4", "--assembly-bond", "32"]
PACKAGE = Path(__file__).resolve().parents[1]


def request(root, kind="run", *extra):
    folder = "reference" if kind == "reference" else "base"
    options = ["--nodes", "4"] if kind == "reference" else NUMERICAL
    return cli.argument_parser().parse_args(
        [kind, *COMMON, *options, "--output", str(root/folder), *extra])


@pytest.fixture(scope="module")
def completed_scan(tmp_path_factory):
    root = tmp_path_factory.mktemp("completed_scan")
    # Real tiny numerical runs supply the same config/snapshot schema as users.
    for kind in ("reference", "run"):
        args = request(root, kind)
        args.function(args)
    return root


@pytest.fixture
def scan(completed_scan, tmp_path):
    root = tmp_path/"scan"
    shutil.copytree(completed_scan, root)
    return root


def digest_tree(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_matching_cache_and_scan_axes(scan):
    assert check_cached(request(scan))
    args = request(scan)
    args.workers, args.blas_threads = 6, 2
    assert check_cached(args)  # placement is not a physical setting
    check_scan(scan, request(scan, "reference"))  # reference levels=6, run levels=2
    assert not check_cached(request(scan/"absent"))


@pytest.mark.parametrize("kind,field,value", [
    ("run", "sigma", .02), ("run", "example", "ising4"),
    ("run", "rotation_ns", 2.), ("run", "steps", 2),
    ("run", "square_native", True), ("run", "max_dt", .05),
    ("run", "heom_bond", 8), ("run", "levels", 4),
    ("run", "reset", "layer"), ("reference", "nodes", 6),
])
def test_changed_config_rejected_without_writes(scan, kind, field, value):
    before = digest_tree(scan)
    args = request(scan, kind)
    setattr(args, field, value)
    with pytest.raises(ValueError, match=field):
        check_cached(args)
    assert digest_tree(scan) == before


@pytest.mark.parametrize("damage", ["unfinished", "missing_field", "corrupt", "missing_tensor"])
def test_unverifiable_old_results_fail_closed(scan, damage):
    folder = scan/"base"
    if damage == "unfinished":
        (folder/"summary.json").rename(folder/"pending.json")
    elif damage == "missing_field":
        for name in ("summary.json", "config.json"):
            data = json.loads((folder/name).read_text())
            (data["config"] if name == "summary.json" else data).pop("sigma")
            (folder/name).write_text(json.dumps(data))
    elif damage == "corrupt":
        (folder/"summary.json").write_text("{")
    else:
        (folder/"spp.npz").rename(folder/"saved.npz")
    with pytest.raises(ValueError, match="Cannot reuse.*new ROOT"):
        check_cached(request(scan))


@pytest.mark.parametrize("input_kind", ["layers", "noise"])
def test_editing_input_file_at_same_path_is_detected(scan, tmp_path, input_kind):
    folder = scan/"base"
    source = tmp_path/f"input_{input_kind}.json"
    shutil.copyfile(folder/f"{input_kind}.json", source)
    args = request(scan, "run", f"--{input_kind}-json", str(source))
    # Recreate the recorded configuration for a run that used this input file.
    for name in ("config.json", "summary.json"):
        data = json.loads((folder/name).read_text())
        config = data["config"] if name == "summary.json" else data
        config[f"{input_kind}_json"] = str(source)
        (folder/name).write_text(json.dumps(data))
    assert check_cached(args)
    if input_kind == "layers":
        data = json.loads(source.read_text())
        data["layers"][0]["blocks"][0]["gates"][0]["angle"] += .1
    else:
        spec = NoiseSpec.load(source)
        data = spec.to_dict()
        data["gwval"] *= 2
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=f"{input_kind}_json content"):
        check_cached(args)


def shell_scan(root, tmp_path, **changes):
    fake_bin = tmp_path/"bin"
    fake_bin.mkdir(exist_ok=True)
    # A submission would be a regression: these checks must run before sbatch.
    fake = fake_bin/"sbatch"
    fake.write_text('#!/bin/sh\nprintf submitted >> "$SUBMISSION_MARKER"\nexit 99\n')
    fake.chmod(0o755)
    env = dict(os.environ, PYTHON=sys.executable, ROOT=str(root), EXAMPLE="one-qubit",
               PULSE_ARGS="--rotation-ns 1", SIGMA=".01", STEPS="1", NODES="4",
               TB="4", HB="4", DT=".1", LV="2", TB_LIST=" ", HB_LIST=" ",
               DT_LIST=" ", LV_LIST=" ", GRID_DT="", GRID_HB="", BASELINES="0",
               REFERENCE="1", SUBMISSION_MARKER=str(tmp_path/"submitted"),
               PATH=str(fake_bin)+os.pathsep+os.environ["PATH"])
    env.pop("LOCAL", None)
    env.update(changes)
    return subprocess.run(["bash", "scripts/scan_convergence.sh"], cwd=PACKAGE,
                          env=env, capture_output=True, text=True, timeout=60)


def test_shell_reuses_matching_scan_without_submission(scan, tmp_path):
    before = digest_tree(scan)
    result = shell_scan(scan, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "done      reference (configuration verified)" in result.stdout
    assert "done      base (configuration verified)" in result.stdout
    assert not (tmp_path/"submitted").exists()
    assert digest_tree(scan) == before


def test_shell_fresh_local_scan_then_reuse(tmp_path):
    root = tmp_path/"fresh"
    first = shell_scan(root, tmp_path, LOCAL="1", WORKERS="1")
    assert first.returncode == 0, first.stdout+first.stderr
    assert "running   reference" in first.stdout and "running   base" in first.stdout
    before = digest_tree(root)
    second = shell_scan(root, tmp_path, LOCAL="1", WORKERS="2")
    assert second.returncode == 0, second.stdout+second.stderr
    assert "configuration verified" in second.stdout
    assert digest_tree(root) == before
    assert not (tmp_path/"submitted").exists()


@pytest.mark.parametrize("changes,field", [
    ({"SIGMA": ".02"}, "sigma"), ({"EXAMPLE": "ising4"}, "example"),
    ({"PULSE_ARGS": "--rotation-ns 2"}, "rotation_ns"),
    ({"NODES": "6"}, "nodes"),
    ({"NODES": "6", "REFERENCE": "0"}, "nodes"),
    ({"HB": "8", "TB_LIST": "8"}, "heom_bond"),
])
def test_shell_rejects_mismatch_before_any_new_work(scan, tmp_path, changes, field):
    before = digest_tree(scan)
    result = shell_scan(scan, tmp_path, **changes)
    assert result.returncode != 0 and field in result.stderr, result.stdout+result.stderr
    assert not (tmp_path/"submitted").exists()
    assert digest_tree(scan) == before


def test_unselected_old_scan_member_cannot_contaminate_new_scan(scan, tmp_path):
    folder = scan/"old-grid-point"
    (scan/"base").rename(folder)
    for name in ("summary.json", "config.json"):
        data = json.loads((folder/name).read_text())
        (data["config"] if name == "summary.json" else data)["sigma"] = .03
        (folder/name).write_text(json.dumps(data))
    result = shell_scan(scan, tmp_path, BASELINES="only")
    assert result.returncode != 0 and "sigma" in result.stderr
    assert not (tmp_path/"submitted").exists()


def test_all_jobs_preflight_before_reference_or_first_submission(scan, tmp_path):
    # A late scan member conflicts, while base and reference do not yet exist.
    (scan/"base").rename(scan/"hb8")
    (scan/"reference").rename(tmp_path/"old-reference")
    before = digest_tree(scan)
    result = shell_scan(scan, tmp_path, HB_LIST="8")
    assert result.returncode != 0 and "heom_bond" in result.stderr
    assert not (tmp_path/"submitted").exists()
    assert digest_tree(scan) == before
