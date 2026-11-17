"""Bath models, independent of circuit layers and pulse Hamiltonians.

Every bath is a list of exponential terms of a correlation function,
C(t) = sum_k gwval_k * exp(-wval_k * t) for t > 0, each carried by one
hierarchy node (Mode). Three kinds of Mode exist:

  commutator  classical real Gaussian noise (real gwval >= 0, real wval):
              one node, raising and lowering both through -i[Q, .].
  left/right  a general (quantum, complex) bath. C(t) is carried by "left"
              nodes lowering through -i g Q rho; C(t)^* by "right" nodes with
              conjugated (g, w) lowering through +i g rho Q. Both raise
              through -i[Q, .]. This is the forward/backward pairing of the
              HEOM-MPS code of Shi et al. (JCP 148, 174102, 2018).

A White term is the delta-correlated remainder k0 of a fit, -k0 [Q,[Q, rho]].

Each term may name a channel; a pulse Segment can rescale the system-bath
coupling of a channel (Q -> gain*Q) while the term is active.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
from .pauli import Z, embed

_ACTIONS = ("commutator", "left", "right")


def _hermitian(operator):
    q = np.array(operator, complex, copy=True)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or not np.allclose(q, q.conj().T):
        raise ValueError("bath coupling operator must be Hermitian")
    return q


@dataclass(frozen=True, eq=False)
class Mode:
    operator: np.ndarray
    gwval: complex
    wval: complex
    levels: int
    action: str = "commutator"
    channel: str = ""
    idle_gain: float = 1.
    scale: complex | None = None  # rho_raw[n] = scale**n sqrt(n!) rho_scaled[n]

    def __post_init__(self):
        g, w = complex(self.gwval), complex(self.wval)
        if (not np.isfinite(g) or not np.isfinite(w) or w.real < 0
                or not isinstance(self.levels, int) or self.levels < 1):
            raise ValueError("invalid mode coefficient, decay rate or levels")
        if self.action not in _ACTIONS:
            raise ValueError(f"mode action must be one of {_ACTIONS}")
        if self.action == "commutator" and (g.imag != 0 or w.imag != 0 or g.real < 0):
            raise ValueError("classical Gaussian modes require real nonnegative gwval/wval")
        if not np.isfinite(self.idle_gain):
            raise ValueError("idle_gain must be finite")
        scale = np.sqrt(abs(g)) if self.scale is None else complex(self.scale)
        if not np.isfinite(scale) or (scale == 0 and g != 0):
            raise ValueError("hierarchy scale must be finite and nonzero")
        object.__setattr__(self, "operator", _hermitian(self.operator))
        object.__setattr__(self, "gwval", g)
        object.__setattr__(self, "wval", w)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def from_exponential(cls, operator, *, gwval, wval, levels, **options):
        """Classical real term gwval*exp(-wval*t); gwval is the variance."""
        return cls(operator, gwval, wval, levels, "commutator", **options)

    @property
    def sigma(self):
        return float(np.sqrt(abs(self.gwval)))

    @property
    def gamma(self):
        return float(self.wval.real)

    def gain(self, segment_gains):
        return dict(segment_gains).get(self.channel, self.idle_gain)

    def up_matrix(self, basis="scaled"):
        """Bath-leg matrix that brings rho[n+1] into the equation of rho[n]."""
        if basis == "scaled":
            return self.scale*np.diag(np.sqrt(np.arange(1, self.levels)), 1)
        if basis == "unscaled":
            return np.diag(np.ones(self.levels-1, complex), 1)
        raise ValueError("unknown hierarchy basis")

    def down_matrix(self, basis="scaled"):
        """Bath-leg matrix that brings rho[n-1] into the equation of rho[n]."""
        if basis == "scaled":
            ratio = 0 if self.scale == 0 else self.gwval/self.scale
            return ratio*np.diag(np.sqrt(np.arange(1, self.levels)), -1)
        if basis == "unscaled":
            return np.diag(self.gwval*np.arange(1, self.levels), -1)
        raise ValueError("unknown hierarchy basis")

    def coupling_matrix(self, basis="scaled"):
        return self.up_matrix(basis)+self.down_matrix(basis)

    def decay_matrix(self):
        return -self.wval*np.diag(np.arange(self.levels)).astype(complex)

    def scaling_factors(self):
        result = np.ones(self.levels, complex)
        for n in range(1, self.levels):
            result[n] = result[n-1]*self.scale*np.sqrt(n)
        return result.real if self.scale.imag == 0 else result


def classical_mode(operator, sigma, gamma, levels, **options):
    """OU (gamma > 0) or static (gamma = 0) Gaussian noise of rms sigma."""
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("invalid mode strength, rate or levels")
    return Mode(operator, sigma**2, gamma, levels, "commutator", **options)


def quantum_modes(operator, wval, gwval, levels, **options):
    """Left nodes for C(t) = sum g exp(-w t), then right nodes for C(t)^*."""
    wval, gwval = np.atleast_1d(wval).astype(complex), np.atleast_1d(gwval).astype(complex)
    if wval.shape != gwval.shape or wval.ndim != 1:
        raise ValueError("one coefficient per decay rate required")
    return (tuple(Mode(operator, g, w, levels, "left", **options) for g, w in zip(gwval, wval))
            + tuple(Mode(operator, g.conjugate(), w.conjugate(), levels, "right", **options)
                    for g, w in zip(gwval, wval)))


@dataclass(frozen=True, eq=False)
class White:
    """Delta-correlated part of a bath: d rho/dt = -rate [Q, [Q, rho]]."""
    operator: np.ndarray
    rate: float
    channel: str = ""
    idle_gain: float = 1.

    def __post_init__(self):
        if not np.isfinite(self.rate) or not np.isfinite(self.idle_gain):
            raise ValueError("white-noise rate and gain must be finite")
        object.__setattr__(self, "operator", _hermitian(self.operator))

    gain = Mode.gain


def load_exponentials(path, *, max_decay=None):
    """Read a barycentric spectrum decomposition table (cfdata_*.dat).

    Lines are `index Re(w) Im(w) Re(g) Im(g)`; a final `-1 k0` line gives the
    white remainder. Only the listed half is returned: quantum_modes adds the
    conjugate partners. max_decay drops terms with Re(w) >= max_decay, which
    are too fast to resolve and should be treated separately.
    Units are those of the file; convert before mixing with ns pulses.
    """
    wval, gwval, k0 = [], [], 0.
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if int(fields[0]) < 0:
            k0 = float(fields[1])
            continue
        w, g = complex(float(fields[1]), float(fields[2])), complex(float(fields[3]), float(fields[4]))
        if max_decay is None or w.real < max_decay:
            wval.append(w)
            gwval.append(g)
    if not wval:
        raise ValueError("no exponential terms read")
    return np.array(wval), np.array(gwval), k0


@dataclass(frozen=True)
class NoiseSpec:
    """Classical dephasing presets that can be stored as noise.json."""
    n_qubits: int
    kind: str = "static"
    sigma: float | tuple[float, ...] = .004
    levels: int = 6
    topology: str = "independent"
    coupled_qubits: tuple[int, ...] | None = None
    modes_per_qubit: int = 1
    gamma: float = .01
    gamma_min: float = 1e-4
    gamma_max: float = .1
    hierarchy_basis: str = "scaled"
    coupling: str = "Z/2"

    def to_dict(self):
        data = asdict(self)
        if self.kind == "static":
            variance = np.asarray(data.pop("sigma"), float)**2
            data["gwval"] = float(variance) if variance.ndim == 0 else variance.tolist()
            data["wval"] = 0.
            for key in ("gamma", "gamma_min", "gamma_max"):
                data.pop(key)
        return dict(schema="heom-noise-v1", **data)

    def save(self, path):
        with Path(path).open("x", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data):
        data = data.copy()
        if data.pop("schema", None) != "heom-noise-v1":
            raise ValueError("unknown noise schema")
        if "wval" in data:
            decay = np.asarray(data.pop("wval"), float)
            if np.any(~np.isfinite(decay)):
                raise ValueError("wval must be finite")
            if data.get("kind", "static") == "static":
                if np.any(decay != 0):
                    raise ValueError("static disorder requires wval=0")
            elif data.get("kind") == "ou" and decay.ndim == 0:
                data["gamma"] = float(decay)
            else:
                raise ValueError("explicit wval input is supported for static or single-rate OU")
        if "gwval" in data:
            if "sigma" in data:
                raise ValueError("specify sigma OR gwval, not both")
            variance = np.asarray(data.pop("gwval"), float)
            if np.any(~np.isfinite(variance)) or np.any(variance < 0):
                raise ValueError("gwval must be a finite nonnegative real variance")
            sigma = np.sqrt(variance)
            data["sigma"] = float(sigma) if sigma.ndim == 0 else tuple(sigma.tolist())
        if isinstance(data.get("sigma"), list):
            data["sigma"] = tuple(data["sigma"])
        if data.get("coupled_qubits") is not None:
            data["coupled_qubits"] = tuple(data["coupled_qubits"])
        return cls(**data)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class NoiseModel:
    modes: tuple[Mode, ...]
    hierarchy_basis: str
    summary: dict


def build_noise(spec):
    """Channels are named z:q<i> (independent) or z:shared for Segment gains."""
    n = spec.n_qubits
    if not isinstance(n, int) or not 1 <= n <= 5 or spec.kind not in ("static", "ou", "one-over-f"):
        raise ValueError("invalid register size or noise kind")
    if spec.coupling not in ("Z/2", "Z"):
        raise ValueError("coupling must be Z/2 or Z")
    factor = .5 if spec.coupling == "Z/2" else 1.
    if spec.hierarchy_basis not in ("scaled", "unscaled"):
        raise ValueError("hierarchy basis must be scaled or unscaled")
    if spec.topology not in ("independent", "shared") or spec.modes_per_qubit < 1:
        raise ValueError("invalid topology or mode count")
    coupled = tuple(range(n)) if spec.coupled_qubits is None else tuple(spec.coupled_qubits)
    if not coupled or len(set(coupled)) != len(coupled) or any(q < 0 or q >= n for q in coupled):
        raise ValueError("invalid coupled-qubit list")
    sigma = np.asarray(spec.sigma, float)
    if sigma.ndim == 0:
        sigma = np.full(n, float(sigma))
    if sigma.shape != (n,) or np.any(~np.isfinite(sigma)) or np.any(sigma < 0):
        raise ValueError("sigma must be scalar or one nonnegative finite value per qubit")
    if spec.kind == "static":
        if spec.levels < 2 and np.any(sigma[list(coupled)] > 0):
            raise ValueError("one static node still needs multiple ADO levels; levels must be >=2")
        if spec.modes_per_qubit != 1:
            raise ValueError("static disorder uses exactly one exponential coordinate per coupled qubit")
        rates = np.array([0.])
    elif spec.kind == "ou":
        if not np.isfinite(spec.gamma) or spec.gamma < 0:
            raise ValueError("invalid OU decay")
        rates = np.full(spec.modes_per_qubit, spec.gamma)
    else:
        if not 0 < spec.gamma_min <= spec.gamma_max or not np.isfinite(spec.gamma_max):
            raise ValueError("positive finite ordered spectrum cutoffs required")
        rates = np.geomspace(spec.gamma_min, spec.gamma_max, spec.modes_per_qubit)
    if spec.topology == "shared":
        if not np.all(sigma[list(coupled)] == sigma[coupled[0]]):
            raise ValueError("shared topology currently requires the same sigma on coupled qubits")
        channels = [(coupled, sum(factor*embed(Z, (q,), n) for q in coupled),
                     sigma[coupled[0]], "z:shared")]
    else:
        channels = [((q,), factor*embed(Z, (q,), n), sigma[q], f"z:q{q}") for q in coupled]
    modes, coordinates = [], []
    for qubits, q, rms, channel in channels:
        gwval = float(rms**2/len(rates))
        for rate in rates:
            mode = Mode.from_exponential(q, gwval=gwval, wval=float(rate),
                                         levels=spec.levels, channel=channel)
            modes.append(mode)
            coordinates.append(dict(qubits=list(qubits), gwval=mode.gwval.real,
                wval=mode.wval.real, levels=mode.levels, coupling=spec.coupling,
                channel=channel))
    return NoiseModel(tuple(modes), spec.hierarchy_basis, dict(
        spec=spec.to_dict(), n_qubits=n, kind=spec.kind, topology=spec.topology,
        coupled_qubits=list(coupled), sigma_per_qubit=sigma.tolist(),
        bath_coordinates=len(modes), coordinates=coordinates,
        hierarchy_basis=spec.hierarchy_basis, stochastic_sampling=False,
        initial_memory="root only; never reset at learning-layer boundaries"))


def static_disorder(n_qubits, *, sigma=None, gwval=None, levels=6, hierarchy_basis="scaled"):
    if sigma is not None and gwval is not None:
        raise ValueError("specify sigma OR gwval")
    if gwval is not None:
        data = dict(schema="heom-noise-v1", n_qubits=n_qubits, kind="static",
                    gwval=gwval, levels=levels, hierarchy_basis=hierarchy_basis)
        spec = NoiseSpec.from_dict(data)
    else:
        spec = NoiseSpec(n_qubits, sigma=.004 if sigma is None else sigma,
                         levels=levels, hierarchy_basis=hierarchy_basis)
    return build_noise(spec)
