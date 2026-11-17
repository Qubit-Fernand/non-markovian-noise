"""Show the sqisw-decomposed Ising Trotter step BEFORE propagating anything.

Writes into --output: layers.json, pulse_schedule.json, layers.txt (layer
table), circuit.txt (Qiskit drawing) and timing.png (pulse timing diagram).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from heom_spp.circuit import layer_unitary, to_qiskit
from heom_spp.native import ising_sqisw_layers, ising_trotter_unitary
from heom_spp.pulse import PulseDurations, RotatingFrames, compile_layers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--qubits", type=int, default=4)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--theta-x", type=float, default=.5)
    p.add_argument("--theta-z", type=float, default=-.075)
    p.add_argument("--one-qubit-ns", type=float, default=15.)
    p.add_argument("--two-qubit-ns", type=float, default=20.)
    p.add_argument("--buffer-ns", type=float, default=5.)
    p.add_argument("--detuning-ghz", type=float, default=.4)
    p.add_argument("--tunable", default="1,3", help="zero-based qubits moved in frequency for an exchange gate")
    args = p.parse_args()
    n = args.qubits
    plan = ising_sqisw_layers(n, theta_x=args.theta_x, theta_z=args.theta_z, steps=args.steps)
    total = np.eye(2**n, dtype=complex)
    for layer in plan.layers:
        total = layer_unitary(layer, n)@total
    target = ising_trotter_unitary(n, args.theta_x, args.theta_z, args.steps)
    error = abs(abs(np.trace(target.conj().T@total))-2**n)/2**n
    durations = PulseDurations(rotation_ns=args.one_qubit_ns, sqisw_ns=args.two_qubit_ns,
                               buffer_ns=args.buffer_ns, virtual_z=True)
    frames = RotatingFrames(tuple(int(q) for q in args.tunable.split(",")),
                            2*np.pi*args.detuning_ghz)
    slots, schedule = compile_layers(plan, durations, frames=frames)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"layers.json").write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    (args.output/"pulse_schedule.json").write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    lines = [f"Ising Trotter step, {n} qubits, {args.steps} step(s): theta_x={args.theta_x}, theta_z={args.theta_z}",
             f"native gates: r (XY square pulse, {args.one_qubit_ns:g} ns), sqisw ({args.two_qubit_ns:g} ns), rz (ideal, 0 ns); "
             f"{args.buffer_ns:g} ns idle before every pulse block",
             f"|1 - |Tr(U_target^dag U_circuit)|/d| = {error:.2e}",
             f"total duration {schedule['total_duration_ns']:g} ns", ""]
    for j, (layer, slot, record) in enumerate(zip(plan.layers, slots, schedule["layers"]), start=1):
        counts = {}
        for block in layer.blocks:
            for g in block.gates:
                counts[g.name] = counts.get(g.name, 0)+1
        lines.append(f"Layer {j} [{layer.name}]  {record['start_ns']:g}-{record['end_ns']:g} ns  "
                     + ", ".join(f"{v} {k}" for k, v in counts.items()))
        for block in layer.blocks:
            lines.append("    " + "  ".join(
                g.name + ("" if g.angle is None else f"({g.angle:+.4f})" if g.phase is None
                          else f"(θ={g.angle:.4f}, φ={g.phase:+.4f})")
                + "[" + ",".join(f"Q{q+1}" for q in g.qubits) + "]" for g in block.gates))
    lines += ["", "Rotating-frame changes (tunable qubit joins its partner's frame for the exchange gate):"]
    for e in schedule["frame_changes"]:
        lines.append(f"    t={e['time_ns']:6.1f} ns  Q{e['qubit']+1} {e['change']:5s} frame of Q{e['partner']+1}: "
                     f"Rz({e['frame_angle']:+.3f}), compensation Rz({e['compensation_angle']:+.3f})")
    (args.output/"layers.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))
    try:
        drawing = str(to_qiskit(plan).draw("text", fold=-1))
        (args.output/"circuit.txt").write_text(drawing+"\n", encoding="utf-8")
    except ImportError:
        print("qiskit not installed: circuit.txt skipped")
    from heom_spp.plot import plot_schedule
    plot_schedule(schedule, args.output/"timing.png",
                  title=f"sqisw Ising Trotter step: {schedule['total_duration_ns']:g} ns, "
                        f"{len(plan.layers)} learning layers")


if __name__ == "__main__":
    main()
