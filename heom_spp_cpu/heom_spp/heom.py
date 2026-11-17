"""HEOM engines on the chain [system Liouville leg, one leg per bath Mode].

For Mode k with coupling operator Q, coefficient g, decay w and segment gain
lambda, the unscaled hierarchy is

  d rho_n/dt = L_S rho_n - sum_k n_k w_k rho_n
               - i sum_k lambda_k [Q_k, rho_{n+e_k}]
               + sum_k lambda_k n_k g_k D_k rho_{n-e_k},

  D = -i[Q, .]   (commutator: classical real Gaussian noise)
  D = -i Q .     (left:  term of C(t))
  D = +i . Q     (right: term of C(t)^*, with conjugated g and w)

L_S holds control, drift, Lindblad jumps and white-noise remainders. The
"scaled" basis divides rho_n by prod_k scale_k^n_k sqrt(n_k!).

Three interchangeable engines share this equation:
  HEOM      compressed explicit RK4 on the MPS (slow, simple reference)
  DenseHEOM sparse matrix exponential, exponential memory (small checks only)
  TDVPHEOM  fixed-rank one-site KSL/TDVP with an MPO generator (production)

Vectorisation is column-major, vec(A rho B) = (B^T kron A) vec(rho).
Tensors are linear maps, not states: nothing is ever normalised.
"""
from dataclasses import dataclass
from math import ceil, prod

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from .mps import MPS, Truncation, linear_combination
from .pulse import Frame, Reset
from .tdvp import expand_bonds, ksl_step, pack


def commutator(q):
    d = len(q)
    return -1j*(np.kron(np.eye(d), q)-np.kron(q.T, np.eye(d)))


def left_action(q):
    return -1j*np.kron(np.eye(len(q)), q)


def right_action(q):
    return 1j*np.kron(q.T, np.eye(len(q)))


def double_commutator(q):
    d, q2 = len(q), q@q
    return np.kron(np.eye(d), q2)+np.kron(q2.T, np.eye(d))-2*np.kron(q.T, q)


def dissipator(jump):
    d = len(jump)
    rate = jump.conj().T@jump
    return (np.kron(jump.conj(), jump)
            -(np.kron(np.eye(d), rate)+np.kron(rate.T, np.eye(d)))/2)


def conjugation(u):
    """Superoperator of rho -> u rho u^dagger."""
    return np.kron(u.conj(), u)


_SYSTEM = dict(commutator=commutator, left=left_action, right=right_action)


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


class HEOM:
    def __init__(self, modes, *, max_dt=0.05, truncation=Truncation(), jumps=(),
                 drift=None, whites=(), hierarchy_basis="scaled"):
        if hierarchy_basis not in ("scaled", "unscaled"):
            raise ValueError("unknown hierarchy basis")
        self.hierarchy_basis = hierarchy_basis
        self.modes = tuple(modes)
        if not self.modes:
            raise ValueError("at least one mode required (zero coupling gives no noise)")
        self.d = len(self.modes[0].operator)
        self.whites = tuple(whites)
        if any(m.operator.shape != (self.d, self.d) for m in self.modes+self.whites):
            raise ValueError("bath operator dimensions differ")
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

    def system_generator(self, segment):
        if segment.dim != self.d:
            raise ValueError("control dimension mismatch")
        out = commutator(segment.hamiltonian+self.drift)+self.markov
        for white in self.whites:
            out = out-white.rate*white.gain(segment.gains)**2*double_commutator(white.operator)
        return out

    def terms(self, segment):
        """Generator as a sum of product operators {site: matrix}."""
        basis = self.hierarchy_basis
        terms = [{0: self.system_generator(segment)}]
        for site, mode in enumerate(self.modes, start=1):
            gain = mode.gain(segment.gains)
            if gain != 0:
                if mode.action == "commutator":
                    terms.append({0: commutator(mode.operator),
                                  site: gain*mode.coupling_matrix(basis)})
                else:
                    terms.append({0: commutator(mode.operator), site: gain*mode.up_matrix(basis)})
                    terms.append({0: _SYSTEM[mode.action](mode.operator),
                                  site: gain*mode.down_matrix(basis)})
            terms.append({site: mode.decay_matrix()})
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

    def _check(self, state):
        if state.dims != (self.d**2, *self.memory_dims):
            raise ValueError("wrong hierarchy shape")

    def reset_bath(self, state):
        """Keep the root (reduced system) component, return to zero ADOs."""
        tail = np.ones((1, 1), complex)
        for core in reversed(state.cores[1:]):
            tail = core[:, 0, :]@tail
        return MPS([state.cores[0]@tail, *self.root().cores])

    def propagate(self, state, slot):
        self._check(state)
        for segment in slot.segments:
            if isinstance(segment, Reset):
                state = self.reset_bath(state)
                continue
            if isinstance(segment, Frame):
                state = state.local(0, conjugation(segment.unitary))
                continue
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
        """Gate-after probe: input U^dagger P_b U, output Tr(P_b . )/d.

        U is the slot's logical ideal unitary, so this is V = G o U^{-1}.
        All bath legs stay OPEN: the result is an MPS over the ADO indices.
        """
        if memory.dims != self.memory_dims:
            raise ValueError("memory dimensions mismatch")
        u = slot.ideal
        probe = (u.conj().T@pauli@u).reshape(-1, order="F")
        state = MPS([probe.reshape(1, -1, 1), *memory.cores])
        evolved = self.propagate(state, slot)
        boundary = np.einsum("s,lsr->lr", pauli.reshape(-1, order="F").conj()/self.d,
                             evolved.cores[0])
        first = np.einsum("ij,jar->iar", boundary, evolved.cores[1])
        self.stats.responses += 1
        return MPS([first, *evolved.cores[2:]])


class DenseHEOM(HEOM):
    """Independent sparse-matrix exponential reference; exponential memory.

    It shares the equation but not the MPS time-stepper. Guarded for small
    verification jobs, never used implicitly by the compressed engines.
    """
    def __init__(self, *args, max_entries=200_000, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_entries = max_entries
        if self.d**2*prod(self.memory_dims) > max_entries:
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
        self._check(state)
        vector = state.dense(self.max_entries).reshape(-1)
        for segment in slot.segments:
            if isinstance(segment, Reset):
                kept = vector.reshape(self.d**2, -1)[:, 0].copy()
                vector = np.zeros_like(vector)
                vector.reshape(self.d**2, -1)[:, 0] = kept
                continue
            if isinstance(segment, Frame):
                vector = (conjugation(segment.unitary)
                          @vector.reshape(self.d**2, -1)).reshape(-1)
                continue
            vector = expm_multiply(segment.duration*self.generator(segment), vector)
            self.stats.steps += 1
        return MPS.from_dense(vector.reshape(self.d**2, *self.memory_dims))


class TDVPHEOM(HEOM):
    """Fixed-rank one-site KSL/TDVP (or compressed RK4) with cached generators.

    Modes sharing a coupling operator share MPO channels, so the MPO width is
    2 + (number of distinct system superoperators), not 2 + number of modes.
    Each query starts with one compressed RK4 step that seeds the bond space
    before fixed-rank TDVP; converge both max_dt and the rank.
    """
    def __init__(self, modes, *, integrator="tdvp", krylov_dim=24,
                 krylov_tol=1e-11, max_dt=.05,
                 truncation=Truncation(max_bond=16, cutoff=1e-13), **options):
        super().__init__(modes, max_dt=max_dt, truncation=truncation, **options)
        if integrator not in ("tdvp", "rk4"):
            raise ValueError("integrator must be tdvp or rk4")
        if integrator == "tdvp" and truncation.max_bond is None:
            raise ValueError("one-site TDVP requires an explicit max_bond")
        if krylov_dim < 1 or krylov_tol <= 0:
            raise ValueError("positive Krylov budget and tolerance required")
        self.integrator, self.krylov_dim, self.krylov_tol = integrator, krylov_dim, krylov_tol
        self._term_cache, self._mpo_cache, self._system_cache = {}, {}, {}
        self._packed_cache = {}
        self.trim = True  # set False to keep every MPO channel on every bond
        self.tdvp_stats = dict(tdvp_steps=0, local_exponentials=0, max_krylov_used=0)

    def terms(self, segment):
        if segment.key not in self._term_cache:
            self._term_cache[segment.key] = super().terms(segment)
        return self._term_cache[segment.key]

    def generator_mpo(self, segment):
        """MPO nodes with axes (left, output, input, right).

        Channel 0 passes the identity, the last channel collects finished
        terms, and one channel per distinct (action, Q) carries a system
        superoperator to the bath node that completes it.
        """
        if segment.key in self._mpo_cache:
            return self._mpo_cache[segment.key]
        channels = []

        def channel(action, operator):
            for i, (a, q) in enumerate(channels):
                if a == action and np.array_equal(q, operator):
                    return i+1
            channels.append((action, operator))
            return len(channels)

        routes = [(channel("commutator", m.operator), channel(m.action, m.operator))
                  for m in self.modes]
        width = len(channels)+2
        end = width-1
        d2 = self.d**2
        head = np.zeros((1, d2, d2, width), complex)
        head[0, :, :, 0] = np.eye(d2)
        head[0, :, :, end] = self.system_generator(segment)
        for i, (action, operator) in enumerate(channels, start=1):
            head[0, :, :, i] = _SYSTEM[action](operator)
        nodes = [head]
        for i, (mode, (up, down)) in enumerate(zip(self.modes, routes)):
            dim, gain = mode.levels, mode.gain(segment.gains)
            w = np.zeros((width, dim, dim, width), complex)
            for c in range(width):
                w[c, :, :, c] = np.eye(dim)
            w[0, :, :, end] = mode.decay_matrix()
            w[up, :, :, end] += gain*mode.up_matrix(self.hierarchy_basis)
            w[down, :, :, end] += gain*mode.down_matrix(self.hierarchy_basis)
            nodes.append(w[:, :, :, end:end+1] if i == len(self.modes)-1 else w)
        if self.trim:
            # Exact: a channel is dropped on every bond after its last use.
            # The identity channel only serves later decay terms, so undamped
            # (static) modes do not carry it at all.
            keep = [None]*len(self.modes)
            alive = np.zeros(width, bool)
            alive[end] = True
            for i in range(len(self.modes), 0, -1):
                alive = alive | np.any(nodes[i][:, :, :, -1] != 0, axis=(1, 2))
                keep[i-1] = np.flatnonzero(alive)
            nodes[0] = nodes[0][:, :, :, keep[0]]
            for i in range(1, len(self.modes)):
                nodes[i] = nodes[i][keep[i-1]][:, :, :, keep[i]]
            nodes[-1] = nodes[-1][keep[-1]]
        nodes = [np.ascontiguousarray(w) for w in nodes]
        self._mpo_cache[segment.key] = nodes
        self._packed_cache[segment.key] = [pack(w) for w in nodes]
        return nodes

    def system_channels(self, segment):
        """System head split per channel, sparse where that pays off."""
        if segment.key not in self._system_cache:
            head = self.generator_mpo(segment)[0][0]
            channels = []
            for j in range(head.shape[-1]):
                matrix = np.ascontiguousarray(head[:, :, j])
                nonzero = np.count_nonzero(matrix)
                channels.append(None if not nonzero else
                    sparse.csr_matrix(matrix) if nonzero < .2*matrix.size else matrix)
            self._system_cache[segment.key] = channels
        return self._system_cache[segment.key]

    def prepare(self, slots):
        """Build generator caches once (per process) before any response."""
        for slot in slots:
            for segment in slot.segments:
                if isinstance(segment, (Frame, Reset)):
                    continue
                if self.integrator == "rk4":
                    self.terms(segment)
                else:
                    self.generator_mpo(segment)
                    self.system_channels(segment)

    def _mpo_warm_step(self, state, operators, channels, dt):
        """One RK4 step of the full MPO action, compressed after each stage."""
        def apply(a):
            cores = []
            for site, (w, c) in enumerate(zip(operators, a.cores)):
                if site == 0:
                    blocks = [np.zeros_like(c[0]) if matrix is None else matrix.dot(c[0])
                              for matrix in channels]
                    cores.append(np.stack(blocks, axis=1).reshape(1, c.shape[1], -1))
                else:
                    b = np.einsum("woiz,bic->wbozc", w, c)
                    cores.append(b.reshape(w.shape[0]*c.shape[0], w.shape[1],
                                           w.shape[3]*c.shape[2]))
            result, errors = MPS(cores).round(self.truncation)
            self.stats.record(result, errors)
            return result
        k1 = apply(state)
        k2 = apply(self._combine([state, k1], [1, dt/2]))
        k3 = apply(self._combine([state, k2], [1, dt/2]))
        k4 = apply(self._combine([state, k3], [1, dt]))
        result = self._combine([state, k1, k2, k3, k4],
                               [1, dt/6, dt/3, dt/3, dt/6])
        self.stats.steps += 1
        return result

    def propagate(self, state, slot):
        if self.integrator == "rk4":
            return super().propagate(state, slot)
        self._check(state)
        cores = None
        for segment in slot.segments:
            if isinstance(segment, Reset):
                # Back to a product state: the next segment warm-starts again.
                state, cores = self.reset_bath(state if cores is None else MPS(cores)), None
                continue
            if isinstance(segment, Frame):
                frame = conjugation(segment.unitary)
                if cores is None:
                    state = state.local(0, frame)
                else:
                    cores[0] = np.einsum("ab,lbr->lar", frame, cores[0])
                continue
            steps = ceil(segment.duration/self.max_dt)
            dt = segment.duration/steps
            first = 0
            operators = self.generator_mpo(segment)
            channels = self.system_channels(segment)
            if cores is None:
                warm = self._mpo_warm_step(state, operators, channels, dt)
                cores = expand_bonds(warm.cores, self.truncation.max_bond)
                first = 1
            for _ in range(first, steps):
                cores, info = ksl_step(cores, operators, dt,
                                      krylov_dim=self.krylov_dim, krylov_tol=self.krylov_tol,
                                      system_channels=channels,
                                      packed=self._packed_cache[segment.key])
                self.stats.steps += 1
                self.tdvp_stats["tdvp_steps"] += 1
                self.tdvp_stats["local_exponentials"] += info["local_exponentials"]
                self.tdvp_stats["max_krylov_used"] = max(
                    self.tdvp_stats["max_krylov_used"], info["max_krylov_used"])
        result = state if cores is None else MPS(cores)
        self.stats.record(result, [])
        return result
