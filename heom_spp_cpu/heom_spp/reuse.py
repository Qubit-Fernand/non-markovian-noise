"""Read-only preflight for scan_convergence.sh; never repair or overwrite runs."""
import argparse
import json
from pathlib import Path

from .bath import NoiseSpec
from .circuit import CircuitLayers
from .cli import argument_parser


# Execution placement does not define the physical/numerical result. Everything
# else is compared, including parser defaults, so old/incomplete schemas fail closed.
_EXECUTION = {"function", "output", "workers", "blas_threads"}
# A scan deliberately varies these numerical settings and reset baselines.
# All remaining model arguments must agree across the entire scan directory.
_SCAN_AXES = {"levels", "max_dt", "heom_bond", "heom_cutoff", "time_bond",
              "time_cutoff", "assembly_bond", "assembly_cutoff", "assembly",
              "krylov_dim", "krylov_tol", "engine", "reset", "chunk_size",
              "nodes", "compression_cutoff", "dense_max_entries",
              "positivity_max_entries"}


def _read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _input_checks(args, folder, *, shared_only):
    # A filename alone cannot detect edits to an input JSON between runs.
    if args.layers_json is not None:
        current = CircuitLayers.load(args.layers_json)
        if args.slot_limit is not None:
            current = CircuitLayers(current.n_qubits, current.layers[:args.slot_limit],
                                    current.source, current.convention)
        saved = CircuitLayers.load(folder/"layers.json")
        if current.operation_signature() != saved.operation_signature():
            raise ValueError("layers_json content differs from saved layers.json")
    if args.noise_json is not None:
        current = NoiseSpec.load(args.noise_json).to_dict()
        saved = NoiseSpec.load(folder/"noise.json").to_dict()
        if shared_only:
            current.pop("levels", None)
            saved.pop("levels", None)
        # Normalise tuple/list representations through JSON.
        if json.loads(json.dumps(current)) != json.loads(json.dumps(saved)):
            raise ValueError("noise_json content differs from saved noise.json")


def check_cached(args, folder=None, *, shared_only=False):
    """Return False for an absent run, True for a verified completed run.

    shared_only compares the physical configuration of any scan member against
    the requested reference, allowing different convergence axes/reset policies.
    """
    folder = Path(folder if folder is not None else args.output)
    if not folder.exists():
        return False
    try:
        if not (folder/"summary.json").is_file():
            raise ValueError("unfinished run (no summary.json); wait for it to finish")
        report = _read(folder/"summary.json")
        if report.get("status") != "complete":
            raise ValueError("summary.json does not report status=complete")
        saved = report["config"]
        if not isinstance(saved, dict):
            raise ValueError("summary config must be an object")
        if (folder/"config.json").exists() and _read(folder/"config.json") != saved:
            raise ValueError("config.json disagrees with summary.json")
        if not shared_only and (report.get("kind") == "reference") != (
                args.function.__name__ == "reference"):
            raise ValueError("run/reference kind differs")
        ignored = _EXECUTION | (_SCAN_AXES if shared_only else set())
        requested = {k: str(v) if isinstance(v, Path) else v
                     for k, v in vars(args).items() if k not in ignored}
        different = [k for k, v in requested.items() if k not in saved or saved[k] != v]
        if different:
            raise ValueError("incompatible configuration fields: " + ", ".join(sorted(different)))
        for name in ("spp.npz", "layers.json", "noise.json"):
            if not (folder/name).is_file():
                raise ValueError(f"completed run is missing {name}")
        _input_checks(args, folder, shared_only=shared_only)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Cannot reuse {folder}: {exc}. Use a new ROOT/output directory; "
                         "existing results were not changed.") from exc
    return True


def check_scan(root, reference_args):
    """Check even unselected old scan members before any work is launched."""
    root = Path(root)
    # Also check an existing reference when REFERENCE=0: converge would still
    # automatically discover it later. Changing NODES must not silently reuse it.
    check_cached(reference_args, root/"reference")
    if root.exists():
        for folder in sorted(root.iterdir()):
            if folder.name != "reference" and folder.is_dir() and any(
                    (folder/name).exists() for name in ("summary.json", "config.json")):
                check_cached(reference_args, folder, shared_only=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    if not options.command or options.command[0] not in ("run", "reference"):
        parser.error("expected run/reference followed by its normal CLI arguments")
    args = argument_parser().parse_args(options.command)
    try:
        if options.scan_root is not None:
            if options.command[0] != "reference":
                parser.error("scan-root requires the requested reference configuration")
            check_scan(options.scan_root, args)
        else:
            check_cached(args)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
