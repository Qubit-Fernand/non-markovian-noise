"""Create a logical learning-layer file from the HTML template or Qiskit QPY."""
import argparse
from pathlib import Path
from heom_spp.circuit import html_ising_layers, from_qiskit, to_qiskit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--qubits", type=int, default=4)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--theta-x", type=float, default=.5)
    p.add_argument("--theta-z", type=float, default=-.075)
    p.add_argument("--input-qpy", type=Path)
    p.add_argument("--layer-ends", help="comma-separated cumulative gate counts, excluding barriers")
    p.add_argument("--output-qpy", type=Path)
    p.add_argument("--text-output", type=Path)
    args = p.parse_args()
    if args.input_qpy:
        from qiskit import qpy
        with args.input_qpy.open("rb") as f:
            circuits = qpy.load(f)
        if len(circuits) != 1:
            p.error("QPY input must contain exactly one circuit")
        ends = [int(x) for x in args.layer_ends.split(",")] if args.layer_ends else None
        plan = from_qiskit(circuits[0], layer_end_indices=ends)
    else:
        if args.layer_ends:
            p.error("layer-ends applies only to QPY input")
        plan = html_ising_layers(args.qubits, theta_x=args.theta_x,
                                 theta_z=args.theta_z, steps=args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plan.save(args.output)
    print(plan.describe())
    if args.text_output:
        with args.text_output.open("x", encoding="utf-8") as f:
            f.write(plan.describe()+"\n")
    if args.output_qpy:
        from qiskit import qpy
        with args.output_qpy.open("xb") as f:
            qpy.dump(to_qiskit(plan), f)


if __name__ == "__main__":
    main()
