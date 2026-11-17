"""Logical gates, parallel blocks, and explicit learning-layer boundaries.

No bath, Hamiltonian, pulse duration, or Qiskit dependency is needed for the
JSON representation. Parallel gate blocks are not automatically twirl slots.
The ideal unitary of a layer is defined HERE, from the logical gates; it never
depends on how a pulse model implements them.
"""
from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from .pauli import I, X, Y, Z, embed

# Logical gates h x y z id rx ry rz cx cz rzz, plus the pulse-level native set
# used after decomposition (heom_spp.native):
#   r      one square XY pulse, exp(-i angle/2 (cos(phase) X + sin(phase) Y)),
#          identical to Qiskit's RGate(theta=angle, phi=phase)
#   rz     ideal Z rotation
#   sqisw  exp(-i pi/8 (XX+YY)), i.e. H = g/2 (XX+YY) for g*T = pi/4. This is
#          the sign convention of HEOM_two-qubit-gate; it is the inverse of
#          the textbook sqrt(iSWAP) = exp(+i pi/8 (XX+YY)).
_ARITY = dict(h=1, x=1, y=1, z=1, id=1, rx=1, ry=1, rz=1, r=1,
              cx=2, cz=2, rzz=2, sqisw=2)
SQISW = expm(-.125j*np.pi*(np.kron(X, X)+np.kron(Y, Y)))
_FIXED = dict(h=(X+Z)/np.sqrt(2), x=X, y=Y, z=Z, id=I,
              cx=np.kron((I+Z)/2, I)+np.kron((I-Z)/2, X),
              cz=np.diag([1, 1, 1, -1]).astype(complex), sqisw=SQISW)
_AXIS = dict(rx=X, ry=Y, rz=Z, rzz=np.kron(Z, Z))


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple[int, ...]
    angle: float | None = None
    phase: float | None = None  # drive phase of an "r" pulse only

    def __post_init__(self):
        object.__setattr__(self, "qubits", tuple(self.qubits))
        if self.name not in _ARITY or len(self.qubits) != _ARITY[self.name]:
            raise ValueError(f"unsupported gate or wrong arity: {self.name}")
        if len(set(self.qubits)) != len(self.qubits) or any(
                not isinstance(q, int) or q < 0 for q in self.qubits):
            raise ValueError("gate requires distinct nonnegative qubit indices")
        rotation = self.name in ("rx", "ry", "rz", "rzz", "r")
        if (self.name == "r") != (self.phase is not None) or (
                self.phase is not None and not isfinite(self.phase)):
            raise ValueError("exactly the r pulse carries a finite drive phase")
        if rotation and (self.angle is None or not isfinite(self.angle)):
            raise ValueError("a rotation needs a bound finite angle")
        if not rotation and self.angle is not None:
            raise ValueError("nonrotation gate must not have an angle")


@dataclass(frozen=True)
class ParallelBlock:
    gates: tuple[Gate, ...]

    def __post_init__(self):
        object.__setattr__(self, "gates", tuple(self.gates))
        qubits = [q for g in self.gates for q in g.qubits]
        if not self.gates or len(set(qubits)) != len(qubits):
            raise ValueError("parallel gates must be nonempty and disjoint")


@dataclass(frozen=True)
class Layer:
    name: str
    blocks: tuple[ParallelBlock, ...]
    step: int = 0

    def __post_init__(self):
        object.__setattr__(self, "blocks", tuple(self.blocks))
        if not self.blocks or self.step < 0:
            raise ValueError("a layer needs blocks and a nonnegative Trotter index")


@dataclass(frozen=True)
class CircuitLayers:
    n_qubits: int
    layers: tuple[Layer, ...]
    source: str = "explicit"
    convention: str = "gate-after"

    def __post_init__(self):
        object.__setattr__(self, "layers", tuple(self.layers))
        if self.n_qubits < 1 or not self.layers or self.convention != "gate-after":
            raise ValueError("nonempty gate-after circuit required")
        if any(q >= self.n_qubits for layer in self.layers for block in layer.blocks
               for gate in block.gates for q in gate.qubits):
            raise ValueError("gate outside the circuit register")

    def to_dict(self):
        return dict(schema="heom-circuit-layers-v1", qubit_order="q0-most-significant", **asdict(self))

    @classmethod
    def from_dict(cls, data):
        if data.get("schema") != "heom-circuit-layers-v1":
            raise ValueError("unknown circuit-layer schema")
        if data.get("qubit_order", "q0-most-significant") != "q0-most-significant":
            raise ValueError("layer JSON must use q0-most-significant operator order")
        layers = [Layer(x["name"], tuple(ParallelBlock(tuple(
            Gate(g["name"], tuple(g["qubits"]), g.get("angle"), g.get("phase"))
            for g in block["gates"])) for block in x["blocks"]), x.get("step", 0))
            for x in data["layers"]]
        return cls(data["n_qubits"], tuple(layers), data.get("source", "json"),
                   data.get("convention", "gate-after"))

    def save(self, path):
        with Path(path).open("x", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def describe(self):
        lines = []
        for j, layer in enumerate(self.layers, start=1):
            blocks = []
            for block in layer.blocks:
                gates = []
                for g in block.gates:
                    label = g.name + ("" if g.angle is None else f"({g.angle:.4g})"
                        if g.phase is None else f"({g.angle:.4g},{g.phase:.4g})")
                    gates.append(label+"["+",".join(f"q{q}" for q in g.qubits)+"]")
                blocks.append(" || ".join(gates))
            lines.append(f"L{j} step={layer.step}: {layer.name}: "+" -> ".join(blocks))
        return "\n".join(lines)


def gate_unitary(gate):
    """Logical unitary on the gate's own qubits, first listed qubit leftmost."""
    if gate.name in _FIXED:
        return _FIXED[gate.name].copy()
    if gate.name == "r":
        return expm(-.5j*gate.angle*(np.cos(gate.phase)*X+np.sin(gate.phase)*Y))
    return expm(-.5j*gate.angle*_AXIS[gate.name])


def layer_unitary(layer, n_qubits):
    """Ideal U_j of one learning layer; blocks act in list order."""
    u = np.eye(2**n_qubits, dtype=complex)
    for block in layer.blocks:
        for gate in block.gates:  # disjoint, so the order inside a block is free
            u = embed(gate_unitary(gate), gate.qubits, n_qubits) @ u
    return u


def html_ising_layers(n_qubits=4, *, theta_x=.5, theta_z=-.075, steps=1):
    """HTML convention: middle Rz is at the END of layer 3.

    n=4 has exactly the four requested learning layers. The analogous open
    chain can use other register sizes; n=2 has only the even-edge two layers.
    """
    if n_qubits < 2 or steps < 1:
        raise ValueError("Ising chain needs at least two qubits and one step")
    even = tuple((q, q+1) for q in range(0, n_qubits-1, 2))
    odd = tuple((q, q+1) for q in range(1, n_qubits-1, 2))
    def block(name, pairs, angle=None):
        return ParallelBlock(tuple(Gate(name, pair, angle) for pair in pairs))
    one = [
        ("rx-cx-even", (block("rx", [(q,) for q in range(n_qubits)], theta_x),
                        block("cx", even))),
        ("rz-cx-even", (block("rz", [(t,) for _, t in even], theta_z),
                        block("cx", even)))]
    if odd:
        one.extend([
            ("cx-rz-middle", (block("cx", odd),
                              block("rz", [(t,) for _, t in odd], theta_z))),
            ("cx-middle", (block("cx", odd),))])
    return CircuitLayers(n_qubits, tuple(Layer(name, blocks, step)
        for step in range(steps) for name, blocks in one), "html-ising")


def one_qubit_layers(steps=1):
    if steps < 1:
        raise ValueError("positive steps required")
    definitions = (("rx", .5, "rx"), ("rz", .4, "rz"), ("rx", -.4, "rx-minus"))
    return CircuitLayers(1, tuple(Layer(label, (ParallelBlock((Gate(name, (0,), angle),)),), step)
        for step in range(steps) for name, angle, label in definitions), "one-qubit")


def _qiskit_sqisw():
    from qiskit.circuit.library import XXPlusYYGate
    return XXPlusYYGate(np.pi/2, 0., label="sqisw")  # exp(-i pi/8 (XX+YY))


def to_qiskit(plan):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(plan.n_qubits)
    for layer in plan.layers:
        for i, block in enumerate(layer.blocks):
            for g in block.gates:
                if g.name == "sqisw":
                    qc.append(_qiskit_sqisw(), list(g.qubits))
                elif g.name == "r":
                    qc.r(g.angle, g.phase, *g.qubits)
                else:
                    method = getattr(qc, g.name)
                    method(g.angle, *g.qubits) if g.angle is not None else method(*g.qubits)
            if i+1 < len(layer.blocks):
                qc.barrier(label="heom:block")
        qc.barrier(label=f"heom:layer:{layer.step}:{layer.name}")
    qc.metadata = dict(heom_source=plan.source, heom_convention="gate-after")
    return qc


def from_qiskit(circuit, *, layer_end_indices=None):
    """Explicit barriers define slots; heom:block barriers define only blocks.

    Without learning barriers the entire circuit is one layer. Only adjacent
    disjoint instructions are coalesced; no gates are commuted or optimized.
    Qubit labels are preserved, while matrices in our compiler use q0-MSB.
    Measurement, classical control, unbound parameters, and other gates fail.
    """
    ends = None
    if layer_end_indices is not None:
        if any(x.operation.name == "barrier" for x in circuit.data):
            raise ValueError("use explicit layer ends OR circuit barriers, not both")
        ends = tuple(layer_end_indices)
        if (not ends or any(not isinstance(x, int) for x in ends)
                or tuple(sorted(set(ends))) != ends or ends[0] < 1
                or ends[-1] != len(circuit.data)):
            raise ValueError("layer ends must be ordered cumulative gate counts ending at circuit length")
    metadata = circuit.metadata or {}
    if metadata.get("heom_convention", "gate-after") != "gate-after":
        raise ValueError("only gate-after convention is supported")
    seen = 0
    layers, blocks, pending, occupied = [], [], [], set()
    def flush_block():
        if pending:
            blocks.append(ParallelBlock(tuple(pending)))
            pending.clear()
            occupied.clear()
    def flush_layer(name=None, step=0):
        flush_block()
        if blocks:
            layers.append(Layer(name or f"layer-{len(layers)+1}", tuple(blocks), step))
            blocks.clear()
    for instruction in circuit.data:
        op = instruction.operation
        sites = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        if op.name == "barrier":
            if set(sites) != set(range(circuit.num_qubits)):
                raise ValueError("learning/block barriers must span the whole register")
            label = op.label or ""
            if label == "heom:block":
                flush_block()
            elif label.startswith("heom:layer:"):
                _, _, step, name = label.split(":", 3)
                flush_layer(name, int(step))
            else:
                flush_layer(label or None)
            continue
        if instruction.clbits or getattr(op, "condition", None) is not None:
            raise ValueError("only unconditional unitary gates are accepted")
        name, angle, phase = op.name, float(op.params[0]) if op.params else None, None
        if name == "xx_plus_yy":
            if not np.allclose([float(x) for x in op.params], [np.pi/2, 0.]):
                raise ValueError("only XXPlusYY(pi/2, 0) = sqisw is a native exchange gate")
            name, angle = "sqisw", None
        elif name == "r":
            phase = float(op.params[1])
        if name not in _ARITY:
            raise ValueError(f"unsupported Qiskit instruction: {op.name}; decompose explicitly")
        gate = Gate(name, sites, angle, phase)
        if occupied.intersection(sites):
            flush_block()
        pending.append(gate)
        occupied.update(sites)
        seen += 1
        if ends is not None and seen in ends:
            flush_layer()
    flush_layer()
    metadata = circuit.metadata or {}
    return CircuitLayers(circuit.num_qubits, tuple(layers), metadata.get("heom_source", "qiskit"))
