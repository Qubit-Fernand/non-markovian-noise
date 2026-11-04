"""Classical Gaussian Hermite hierarchy, Eq. (HEOM) of note.tex.

System Liouville leg first (column-major vec), then one Hermite-index leg per
independent OU/static mode. Per-mode cutoff retains k=0,...,levels-1. MPS
propagation applies a sum of product operators, using compressed explicit RK4;
this is not TDVP and is not the oscillator-bath HEOM in nmnoise.quantum.
"""
from dataclasses import dataclass
from math import ceil, prod

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from .mps import K, MPS, Truncation, linear_combination


def commutator(q):
    d = len(q)
    return -1j * (np.kron(np.eye(d), q) - np.kron(q.T, np.eye(d)))


def dissipator(jump):
    d = len(jump)
    rate = jump.conj().T @ jump
    return (np.kron(jump.conj(), jump)
            - (np.kron(np.eye(d), rate) + np.kron(rate.T, np.eye(d)))/2)


@dataclass(frozen=True)
class Mode:
    operator: np.ndarray
    sigma: float
    gamma: float
    levels: int

    def __post_init__(self):
        q = np.array(self.operator, complex, copy=True)
        if q.ndim != 2 or q.shape[0] != q.shape[1] or not np.allclose(q, q.conj().T):
            raise ValueError("mode operator must be Hermitian")
        if (not np.isfinite(self.sigma) or not np.isfinite(self.gamma)
                or min(self.sigma, self.gamma) < 0 or self.levels < 1
                or not isinstance(self.levels, int)):
            raise ValueError("invalid mode strength, rate or levels")
        object.__setattr__(self, "operator", q)


@dataclass(frozen=True)
class Segment:
    hamiltonian: np.ndarray
    duration: float

    def __post_init__(self):
        h = np.array(self.hamiltonian, complex, copy=True)
        if h.ndim != 2 or h.shape[0] != h.shape[1] or not np.allclose(h, h.conj().T):
            raise ValueError("control Hamiltonian must be Hermitian")
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("positive finite duration required")
        object.__setattr__(self, "hamiltonian", h)


@dataclass(frozen=True)
class Slot:
    segments: tuple[Segment, ...]
    name: str = "slot"

    def __post_init__(self):
        if not self.segments or any(s.hamiltonian.shape != self.segments[0].hamiltonian.shape
                                    for s in self.segments):
            raise ValueError("slot needs nonempty, dimension-matched segments")

    @property
    def ideal(self):
        u = np.eye(len(self.segments[0].hamiltonian), dtype=complex)
        for seg in self.segments:
            u = expm(-1j * seg.duration * seg.hamiltonian) @ u
        return u


@dataclass
class Diagnostics:
    steps: int = 0
    responses: int = 0
    max_bond_seen: int = 1
    svd_discarded_norm_sum: float = 0.0
    svd_discarded_norm_max: float = 0.0

    def record(self, state, errors):
        self.max_bond_seen = max(self.max_bond_seen, max(state.bonds, default=1))
        self.svd_discarded_norm_sum += sum(errors)
        self.svd_discarded_norm_max = max(self.svd_discarded_norm_max, max(errors, default=0.0))


class HermiteHEOM:
    def __init__(self, modes, *, max_dt=0.05, truncation=Truncation(), jumps=(),
                 drift=None):
        self.modes = tuple(modes)
        if not self.modes:
            raise ValueError("at least one mode required (sigma=0 gives no noise)")
        self.d = len(self.modes[0].operator)
        if any(m.operator.shape != (self.d, self.d) for m in self.modes):
            raise ValueError("mode dimensions differ")
        if self.d < 2 or self.d & (self.d-1):
            raise ValueError("system dimension must be a power of two")
        self.n = self.d.bit_length()-1
        if not np.isfinite(max_dt) or max_dt <= 0:
            raise ValueError("max_dt must be finite and positive")
        self.max_dt, self.truncation = max_dt, truncation
        self.drift = np.zeros((self.d, self.d), complex) if drift is None else np.asarray(drift, complex)
        if self.drift.shape != (self.d, self.d) or not np.allclose(self.drift, self.drift.conj().T):
            raise ValueError("drift must be a Hermitian system operator")
        self.markov = np.zeros((self.d**2, self.d**2), complex)
        for jump in jumps:
            jump = np.asarray(jump, complex)
            if jump.shape != (self.d, self.d):
                raise ValueError("jump dimension mismatch; include sqrt(rate) in jump")
            self.markov += dissipator(jump)
        self.stats = Diagnostics()

    @property
    def memory_dims(self):
        return tuple(m.levels for m in self.modes)

    def root(self):
        return MPS.product([np.eye(m.levels)[0] for m in self.modes])

    def terms(self, segment):
        if segment.hamiltonian.shape != (self.d, self.d):
            raise ValueError("control dimension mismatch")
        terms = [{0: commutator(segment.hamiltonian + self.drift) + self.markov}]
        for site, mode in enumerate(self.modes, start=1):
            ladder = np.diag(np.sqrt(np.arange(1, mode.levels)), 1)
            terms.append({0: mode.sigma * commutator(mode.operator), site: ladder+ladder.T})
            terms.append({site: -mode.gamma*np.diag(np.arange(mode.levels))})
        return terms

    def _combine(self, states, weights):
        out, errors = linear_combination(states, weights, self.truncation)
        self.stats.record(out, errors)
        return out

    def _rhs(self, state, terms):
        states = []
        for term in terms:
            out = state
            for site, operator in term.items():
                out = out.local(site, operator)
            states.append(out)
        return self._combine(states, [1]*len(states))

    def propagate(self, state, slot):
        if state.dims != (self.d**2, *self.memory_dims):
            raise ValueError("wrong hierarchy shape")
        for segment in slot.segments:
            terms = self.terms(segment)
            steps = ceil(segment.duration/self.max_dt)
            dt = segment.duration/steps
            for _ in range(steps):
                k1 = self._rhs(state, terms)
                k2 = self._rhs(self._combine([state, k1], [1, dt/2]), terms)
                k3 = self._rhs(self._combine([state, k2], [1, dt/2]), terms)
                k4 = self._rhs(self._combine([state, k3], [1, dt]), terms)
                state = self._combine([state, k1, k2, k3, k4], [1, dt/6, dt/3, dt/3, dt/6])
                self.stats.steps += 1
        return state

    def response(self, slot, pauli, memory):
        """T_hat(b)X: input U† P_b U, output trace against P_b / d.

        Applying U† on the input implements V=G o U^{-1} (gate-after).
        All ADO legs remain OPEN. No density-matrix normalization is allowed.
        """
        if memory.dims != self.memory_dims:
            raise ValueError("memory dimensions mismatch")
        u = slot.ideal
        probe = (u.conj().T @ pauli @ u).reshape(-1, order="F")
        state = MPS([probe.reshape(1, -1, 1), *memory.cores])
        evolved = self.propagate(state, slot)
        boundary = K.einsum("s,lsr->lr", pauli.reshape(-1, order="F").conj()/self.d,
                            evolved.cores[0])
        first = K.einsum("ij,jar->iar", boundary, evolved.cores[1])
        self.stats.responses += 1
        return MPS([first, *evolved.cores[2:]])


class DenseHermiteHEOM(HermiteHEOM):
    """Independent sparse-matrix exponential reference; exponential memory.

    It shares the specified equation but not the MPS time-stepper. Guarded for
    small verification jobs, never used implicitly by the compressed engine.
    """
    def __init__(self, *args, max_entries=200_000, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_entries = max_entries
        if self.d**2 * prod(self.memory_dims) > max_entries:
            raise ValueError("dense hierarchy exceeds max_entries")

    def generator(self, segment):
        dims = (self.d**2, *self.memory_dims)
        generator = sparse.csr_matrix((prod(dims), prod(dims)), dtype=complex)
        for term in self.terms(segment):
            operator = sparse.csr_matrix([[1.0]])
            for site, dim in enumerate(dims):
                operator = sparse.kron(operator, term.get(site, sparse.eye(dim)), format="csr")
            generator += operator
        return generator

    def propagate(self, state, slot):
        vector = state.dense(self.max_entries).reshape(-1)
        for segment in slot.segments:
            vector = expm_multiply(segment.duration*self.generator(segment), vector)
            self.stats.steps += 1
        return MPS.from_dense(vector.reshape(self.d**2, *self.memory_dims))
