"""Physical schedule of one learning layer, and a placeholder pulse compiler.

A Slot is what the HEOM engine propagates: timed Segments (control
Hamiltonian + noise evolve together) and instantaneous Frames (virtual gates).
Its ideal unitary is supplied by the logical layer, NOT derived from the
pulses, so that coherent pulse errors and leakage are never absorbed into the
"ideal gate" that twirling compensates.

The compiler below still uses ideal constant-amplitude pulses. Calibrated
envelopes (adiabatic / non-adiabatic CZ, DRAG) belong in this module and only
need to emit Segments/Frames; the engine and the SPP construction do not change.
"""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.linalg import expm
from .circuit import gate_unitary, layer_unitary
from .pauli import I, X, Y, Z, embed


@dataclass(frozen=True, eq=False)
class Segment:
    """Constant control Hamiltonian for `duration`.

    gains rescales the system-bath coupling of named bath channels during this
    segment (Q -> gain*Q), e.g. enhanced flux noise while a qubit is detuned,
    or a coupler bath that is only on during a two-qubit gate. Channels not
    listed use the mode's own idle_gain. ADOs are never reset.
    """
    hamiltonian: np.ndarray
    duration: float
    gains: tuple = ()

    def __post_init__(self):
        h = np.array(self.hamiltonian, complex, copy=True)
        if h.ndim != 2 or h.shape[0] != h.shape[1] or not np.allclose(h, h.conj().T):
            raise ValueError("control Hamiltonian must be Hermitian")
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("positive finite duration required")
        gains = tuple(sorted(dict(self.gains).items()))
        if any(not isinstance(k, str) or not np.isfinite(v) for k, v in gains):
            raise ValueError("gains map channel names to finite factors")
        object.__setattr__(self, "hamiltonian", h)
        object.__setattr__(self, "gains", gains)

    @property
    def dim(self):
        return len(self.hamiltonian)

    @property
    def key(self):
        """Everything that determines the HEOM generator of this segment."""
        return self.hamiltonian.shape, self.hamiltonian.tobytes(), self.gains

    @property
    def propagator(self):
        return expm(-1j*self.duration*self.hamiltonian)


@dataclass(frozen=True, eq=False)
class Frame:
    """Instantaneous, noise-free system unitary (virtual Z / frame update)."""
    unitary: np.ndarray

    def __post_init__(self):
        u = np.array(self.unitary, complex, copy=True)
        if u.ndim != 2 or u.shape[0] != u.shape[1] or not np.allclose(
                u@u.conj().T, np.eye(len(u)), atol=1e-10):
            raise ValueError("frame must be unitary")
        object.__setattr__(self, "unitary", u)

    @property
    def dim(self):
        return len(self.unitary)

    @property
    def propagator(self):
        return self.unitary


@dataclass(frozen=True, eq=False)
class Reset:
    """Bath reset: keep the reduced system state, put every ADO back to zero.

    This is the factorised (Born) initial condition of a gate-level master
    equation: whatever follows starts from an uncorrelated equilibrium bath.
    """
    dim: int

    @property
    def propagator(self):
        return np.eye(self.dim, dtype=complex)


@dataclass(frozen=True, eq=False)
class Slot:
    """One learning layer = one SPP time step.

    unitary is the ideal logical U_j. If omitted it falls back to the product
    of the listed operations, which is only appropriate for hand-written ideal
    pulses (tests, analytic benchmarks).
    """
    segments: tuple
    name: str = "slot"
    unitary: np.ndarray | None = None

    def __post_init__(self):
        segments = tuple(self.segments)
        if not segments or any(not isinstance(s, (Segment, Frame, Reset)) for s in segments):
            raise ValueError("slot needs at least one Segment, Frame or Reset")
        dim = segments[0].dim
        if any(s.dim != dim for s in segments):
            raise ValueError("slot operations must share one system dimension")
        if self.unitary is None:
            u = np.eye(dim, dtype=complex)
            for s in segments:
                u = s.propagator@u
        else:
            u = np.array(self.unitary, complex, copy=True)
            if u.shape != (dim, dim) or not np.allclose(u@u.conj().T, np.eye(dim), atol=1e-10):
                raise ValueError("ideal layer unitary has the wrong shape or is not unitary")
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "unitary", u)

    @property
    def ideal(self):
        return self.unitary

    @property
    def duration(self):
        return sum(s.duration for s in self.segments if isinstance(s, Segment))


@dataclass(frozen=True)
class PulseDurations:
    rotation_ns: float = 20.   # every single-qubit pulse, including "r"
    cx_ns: float = 80.
    cz_ns: float = 80.
    rzz_ns: float = 80.
    sqisw_ns: float = 20.
    idle_ns: float = 20.       # an explicit "id" gate
    buffer_ns: float = 0.      # idle inserted before every pulse block
    # True: rz and z are zero-duration ideal rotations (Frames). False keeps
    # the earlier finite-pulse rz model.
    virtual_z: bool = False

    def __post_init__(self):
        times = {k: v for k, v in asdict(self).items() if k.endswith("_ns")}
        buffer = times.pop("buffer_ns")
        if any(not np.isfinite(t) or t <= 0 for t in times.values()):
            raise ValueError("pulse durations must be positive and finite")
        if not np.isfinite(buffer) or buffer < 0:
            raise ValueError("buffer must be nonnegative and finite")


SQUARE_NATIVE = PulseDurations(rotation_ns=15., sqisw_ns=20., buffer_ns=5., virtual_z=True)


@dataclass(frozen=True)
class RotatingFrames:
    """Frame bookkeeping of flux-tuned exchange gates (HEOM_two-qubit-gate).

    Frame A (idle, single-qubit pulses): every qubit rotates at its own idle
    frequency. Frame B (exchange gate): the tunable qubit of the pair is
    brought to its partner's frequency and both rotate at that frequency.
    Entering B at time t is Rz(-detuning*t) on the tunable qubit, leaving at
    t' is Rz(+detuning*t'). These are exact changes of description, not noise.

    compensate=True: the control software applies the opposite ideal Z at the
    same instant, so the gate seen by the circuit is the plain sqisw and no
    Frame is emitted (both rotations are still listed in the schedule).
    compensate=False: the frame rotations are left in as Frames, i.e. the
    exchange gate carries its uncorrected phase and shows up as a coherent error.
    """
    tunable: tuple
    detuning: float = 2*np.pi*.4  # rad/ns; 0.4 GHz as in HEOM_two-qubit-gate
    compensate: bool = True

    def __post_init__(self):
        object.__setattr__(self, "tunable", tuple(self.tunable))
        if not np.isfinite(self.detuning):
            raise ValueError("detuning must be finite")

    def moved(self, gate):
        moved = [q for q in gate.qubits if q in self.tunable]
        if len(moved) != 1:
            raise ValueError(f"exchange gate on {gate.qubits} needs exactly one tunable qubit")
        return moved[0]


_ROTATION_AXIS = dict(rx=X, ry=Y, rz=Z)
_PI_AXIS = dict(x=X, y=Y, z=Z, h=(X+Z)/np.sqrt(2))


def gate_hamiltonian(gate, n_qubits, durations):
    """Constant (square) pulse reproducing the logical gate up to a global phase.

    Also returns the control amplitudes in rad/ns: H = omega_x/2 X + omega_y/2 Y
    for a drive, H = g/2 (XX+YY) for the exchange gate.
    """
    name, controls = gate.name, {}
    if name == "r":
        duration = durations.rotation_ns
        omega = gate.angle/duration
        controls = dict(omega_x=omega*np.cos(gate.phase), omega_y=omega*np.sin(gate.phase))
        operator = .5*(controls["omega_x"]*X+controls["omega_y"]*Y)
    elif name in _ROTATION_AXIS:
        duration = durations.rotation_ns
        operator = _ROTATION_AXIS[name]*gate.angle/(2*duration)
        if name != "rz":
            controls = {"omega_"+name[1]: gate.angle/duration}
    elif name in _PI_AXIS:
        duration = durations.rotation_ns
        operator = _PI_AXIS[name]*np.pi/(2*duration)
        if name in ("x", "y"):
            controls = {"omega_"+name: np.pi/duration}
    elif name == "sqisw":
        duration = durations.sqisw_ns
        controls = dict(g=np.pi/(4*duration))
        operator = .5*controls["g"]*(np.kron(X, X)+np.kron(Y, Y))
    elif name == "cx":
        duration = durations.cx_ns
        operator = np.pi/duration*np.kron((I-Z)/2, (I-X)/2)
    elif name == "cz":
        duration = durations.cz_ns
        operator = np.pi/duration*np.kron((I-Z)/2, (I-Z)/2)
    elif name == "rzz":
        duration = durations.rzz_ns
        operator = gate.angle/(2*duration)*np.kron(Z, Z)
    elif name == "id":
        duration = durations.idle_ns
        operator = np.zeros((2, 2), complex)
    else:
        raise ValueError(f"no pulse model for {name}")
    return embed(operator, gate.qubits, n_qubits), duration, controls


def _gate_gains(gate, table):
    return {template.format(*gate.qubits): factor
            for template, factor in (table or {}).get(gate.name, {}).items()}


def compile_layers(plan, durations=PulseDurations(), *, gains=None, frames=None,
                   reset_blocks=False):
    """Disjoint gates run simultaneously; shorter pulses finish into idle.

    Each logical learning layer becomes one Slot whose ideal unitary comes
    from the logical gates. Every pulse block is preceded by buffer_ns of
    idle, during which the noise keeps acting.

    gains optionally maps a gate name to {channel template: factor};
    templates are formatted with the gate's qubits, e.g.
    {"cz": {"z:q{1}": 5.0, "coupler:q{0}q{1}": 1.0}}. Factors of
    simultaneously active gates on the same channel multiply.

    frames (RotatingFrames) adds the rotating-frame bookkeeping of sqisw gates.

    reset_blocks=True puts a bath Reset after every pulse block (its buffer
    included): the gate-level independent-noise model.

    The schedule lists every pulse with its amplitudes and every ideal Z, for
    inspection and plotting before anything is propagated.
    """
    n = plan.n_qubits
    if n > 5:
        raise ValueError("dense pulse compiler supports at most 5 qubits; logical layer files can be larger")
    zero = np.zeros((2**n,)*2, complex)
    slots, records, pulse_list, z_list, frame_list = [], [], [], [], []
    clock = 0.
    for j, layer in enumerate(plan.layers):
        operations, events = [], []
        start_of_layer = clock
        for block_index, block in enumerate(layer.blocks):
            virtual = [g for g in block.gates if durations.virtual_z and g.name in ("rz", "z")]
            timed = [g for g in block.gates if g not in virtual]
            if virtual:
                u = np.eye(2**n, dtype=complex)
                for g in virtual:
                    u = embed(gate_unitary(g), g.qubits, n)@u
                    z_list.append(dict(layer=j, time_ns=clock, qubit=g.qubits[0],
                                       angle=np.pi if g.name == "z" else g.angle))
                operations.append(Frame(u))
                events.append(dict(block=block_index, start_ns=clock, end_ns=clock,
                                   virtual_gates=[g.name for g in virtual]))
            if not timed:
                continue
            if durations.buffer_ns:
                operations.append(Segment(zero, durations.buffer_ns))
                events.append(dict(block=block_index, start_ns=clock,
                                   end_ns=clock+durations.buffer_ns, active_gates=[], buffer=True))
                clock += durations.buffer_ns
            pulses = [gate_hamiltonian(g, n, durations) for g in timed]
            exchanges = [(g, t) for g, (_, t, _) in zip(timed, pulses) if g.name == "sqisw"] \
                if frames is not None else []

            def frame_change(sign, when):
                u = np.eye(2**n, dtype=complex)
                for g, t in exchanges:
                    q, time = frames.moved(g), clock+(t if when == "leave" else 0.)
                    angle = sign*frames.detuning*time
                    frame_list.append(dict(layer=j, time_ns=time, qubit=q, partner=[
                        x for x in g.qubits if x != q][0], change=when, frame_angle=angle,
                        compensation_angle=-angle if frames.compensate else 0.))
                    u = embed(expm(-.5j*angle*Z), (q,), n)@u
                if exchanges and not frames.compensate:
                    operations.append(Frame(u))

            frame_change(-1., "enter")
            for g, (_, t, controls) in zip(timed, pulses):
                pulse_list.append(dict(layer=j, block=block_index, gate=g.name,
                    qubits=list(g.qubits), start_ns=clock, end_ns=clock+t,
                    angle=g.angle, phase=g.phase, **controls))
            previous = 0.
            for end in sorted(set(t for _, t, _ in pulses)):
                active = [(g, m) for g, (m, t, _) in zip(timed, pulses) if t > previous]
                h = sum((m for _, m in active), zero)
                factors = {}
                for g, _ in active:
                    for channel, factor in _gate_gains(g, gains).items():
                        factors[channel] = factors.get(channel, 1.)*factor
                operations.append(Segment(h, end-previous, tuple(factors.items())))
                events.append(dict(block=block_index, start_ns=clock+previous,
                    end_ns=clock+end, active_gates=[g.name for g, _ in active],
                    gains=factors))
                previous = end
            frame_change(+1., "leave")
            clock += previous
            if reset_blocks:
                operations.append(Reset(2**n))
                events.append(dict(block=block_index, start_ns=clock, end_ns=clock, bath_reset=True))
        slots.append(Slot(tuple(operations), layer.name, layer_unitary(layer, n)))
        records.append(dict(layer=j, step=layer.step, name=layer.name,
                            segments=events, start_ns=start_of_layer, end_ns=clock))
    return tuple(slots), dict(schema="heom-pulse-schedule-v3", convention="gate-after",
        n_qubits=n, durations=asdict(durations), layers=records, total_duration_ns=clock,
        pulses=pulse_list, ideal_z=z_list, frame_changes=frame_list,
        frames=None if frames is None else asdict(frames),
        units="time ns; omega_x, omega_y, g in rad/ns",
        ideal_unitary="logical layer (independent of the pulse model)",
        pulse_models="square pulses; not device calibrated")


def system_terms(n_qubits, *, t1_ns=0., residual_zz=0.):
    """Residual Hamiltonian and Lindblad jumps; excluded from ideal compensation."""
    if not np.isfinite(t1_ns) or t1_ns < 0 or not np.isfinite(residual_zz):
        raise ValueError("invalid T1 or residual coupling")
    drift = sum((embed(np.kron(Z, Z), (q, q+1), n_qubits)
                 for q in range(n_qubits-1)), np.zeros((2**n_qubits,)*2, complex))*residual_zz
    jumps = tuple(embed(np.array([[0, 1], [0, 0]], complex), (q,), n_qubits)/np.sqrt(t1_ns)
                  for q in range(n_qubits)) if t1_ns else ()
    return drift, jumps
