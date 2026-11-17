"""Construct a circuit-independent classical HEOM noise specification."""
import argparse
from pathlib import Path
import json
import numpy as np
from heom_spp.bath import NoiseSpec, build_noise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--qubits", type=int, default=4)
    p.add_argument("--kind", choices=("static", "ou", "one-over-f"), default="static")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--sigma", type=float, nargs="+")
    group.add_argument("--gwval", type=float, nargs="+", help="static variance(s), alternative to sigma")
    p.add_argument("--levels", type=int, default=6)
    p.add_argument("--coupling", choices=("Z/2", "Z"), default="Z/2")
    p.add_argument("--hierarchy-basis", choices=("scaled", "unscaled"), default="scaled")
    p.add_argument("--topology", choices=("independent", "shared"), default="independent")
    p.add_argument("--modes-per-qubit", type=int)
    p.add_argument("--gamma", type=float, default=.01)
    p.add_argument("--gamma-min", type=float, default=1e-4)
    p.add_argument("--gamma-max", type=float, default=.1)
    args = p.parse_args()
    values = args.sigma or [.004]
    if args.gwval is not None:
        if args.kind != "static":
            p.error("gwval CLI input is for the single-coordinate static model")
        if any(x < 0 or not np.isfinite(x) for x in args.gwval):
            p.error("gwval must be nonnegative and finite")
        values = np.sqrt(args.gwval).tolist()
    spec = NoiseSpec(args.qubits, kind=args.kind,
        sigma=values[0] if len(values) == 1 else tuple(values),
        levels=args.levels, topology=args.topology,
        modes_per_qubit=args.modes_per_qubit or (12 if args.kind == "one-over-f" else 1),
        hierarchy_basis=args.hierarchy_basis, coupling=args.coupling, gamma=args.gamma,
        gamma_min=args.gamma_min, gamma_max=args.gamma_max)
    model = build_noise(spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec.save(args.output)
    print(json.dumps(model.summary, indent=2))


if __name__ == "__main__":
    main()
