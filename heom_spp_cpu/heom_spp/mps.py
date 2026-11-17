"""Open-boundary tensor trains on plain NumPy.

These tensors are linear functions, not quantum wavefunctions: never normalize
their Euclidean norm implicitly. A physical leg may label an ADO or a Pauli.
"""
from dataclasses import dataclass
from math import prod

import numpy as np


@dataclass(frozen=True)
class Truncation:
    max_bond: int | None = None
    cutoff: float = 0.0  # absolute discarded Frobenius norm per SVD

    def __post_init__(self):
        if self.max_bond is not None and (not isinstance(self.max_bond, int) or self.max_bond < 1):
            raise ValueError("max_bond must be positive")
        if not np.isfinite(self.cutoff) or self.cutoff < 0:
            raise ValueError("cutoff must be finite and nonnegative")


def split(matrix, truncation=Truncation()):
    """Truncated SVD. Keeps the fewest singular values whose discarded
    Frobenius norm is <= cutoff, capped by max_bond; exact zeros are dropped.
    Returns u, s, vh and the discarded norm.
    """
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    tail = np.sqrt(np.cumsum(s[::-1]**2))[::-1]  # tail[k] = norm of s[k:]
    keep = int(np.count_nonzero(tail > truncation.cutoff))
    if truncation.max_bond is not None:
        keep = min(keep, truncation.max_bond)
    discarded = float(tail[keep]) if keep < len(s) else 0.0
    if keep == 0:
        return (np.zeros((matrix.shape[0], 1), complex), np.zeros(1),
                np.zeros((1, matrix.shape[1]), complex), discarded)
    return u[:, :keep], s[:keep], vh[:keep], discarded


def _qr(matrix):
    return np.linalg.qr(matrix, mode="reduced")


class MPS:
    """Cores have axes (left bond, physical label, right bond)."""

    def __init__(self, cores):
        self.cores = [np.array(c, dtype=complex, copy=True) for c in cores]
        if not self.cores or any(c.ndim != 3 or min(c.shape) < 1 for c in self.cores):
            raise ValueError("nonempty rank-three cores required")
        if self.cores[0].shape[0] != 1 or self.cores[-1].shape[2] != 1:
            raise ValueError("boundary bonds must equal one")
        if any(a.shape[2] != b.shape[0] for a, b in zip(self.cores, self.cores[1:])):
            raise ValueError("adjacent bonds do not match")
        if any(not np.all(np.isfinite(c)) for c in self.cores):
            raise ValueError("nonfinite tensor")

    @property
    def dims(self):
        return tuple(c.shape[1] for c in self.cores)

    @property
    def bonds(self):
        return tuple(c.shape[2] for c in self.cores[:-1])

    @property
    def storage(self):
        return sum(c.size for c in self.cores)

    @classmethod
    def product(cls, vectors):
        return cls([np.asarray(v).reshape(1, -1, 1) for v in vectors])

    @classmethod
    def from_dense(cls, array, truncation=Truncation()):
        array = np.asarray(array, complex)
        if array.ndim < 1 or array.size == 0:
            raise ValueError("tensor needs at least one nonempty physical axis")
        cores, left = [], 1
        rest = array
        for dim in array.shape[:-1]:
            u, s, vh, _ = split(rest.reshape(left * dim, -1), truncation)
            cores.append(u.reshape(left, dim, len(s)))
            rest, left = s[:, None] * vh, len(s)
        return cls(cores + [rest.reshape(left, array.shape[-1], 1)])

    def dense(self, max_entries=1_000_000):
        if prod(self.dims) > max_entries:
            raise ValueError("dense enumeration exceeds max_entries")
        out = self.cores[0][0]
        for core in self.cores[1:]:
            out = np.tensordot(out, core, axes=(-1, 0))
        return out[..., 0]

    def value(self, labels):
        if len(labels) != len(self.cores):
            raise ValueError("one label per site required")
        out = np.ones(1, complex)
        for a, c in zip(labels, self.cores):
            if not isinstance(a, (int, np.integer)) or not 0 <= a < c.shape[1]:
                raise ValueError("invalid physical label")
            out = out @ c[:, a, :]
        return out.item()

    def total(self):
        out = np.ones(1, complex)
        for c in self.cores:
            out = out @ c.sum(axis=1)
        return out.item()

    def scaled(self, scalar):
        return MPS([self.cores[0] * scalar, *self.cores[1:]])

    def local(self, site, matrix):
        cores = self.cores.copy()
        cores[site] = np.einsum("ab,lbr->lar", np.asarray(matrix), cores[site])
        return MPS(cores)

    def inner(self, other):
        self._same_dims(other)
        env = np.ones((1, 1), complex)
        for a, b in zip(self.cores, other.cores):
            env = np.einsum("ij,iar,jas->rs", env, a.conj(), b)
        return env.item()

    def norm(self):
        """Stable Frobenius norm, including nearly cancelling sums."""
        return float(np.linalg.norm(self.right_canonical().cores[0]))

    def _same_dims(self, other):
        if self.dims != other.dims:
            raise ValueError("physical dimensions differ")

    def add(self, other):
        self._same_dims(other)
        if len(self.cores) == 1:
            return MPS([self.cores[0] + other.cores[0]])
        cores = [np.concatenate((self.cores[0], other.cores[0]), axis=2)]
        for a, b in zip(self.cores[1:-1], other.cores[1:-1]):
            c = np.zeros((a.shape[0]+b.shape[0], a.shape[1], a.shape[2]+b.shape[2]), complex)
            c[:a.shape[0], :, :a.shape[2]] = a
            c[a.shape[0]:, :, a.shape[2]:] = b
            cores.append(c)
        return MPS(cores + [np.concatenate((self.cores[-1], other.cores[-1]), axis=0)])

    def hadamard(self, other):
        self._same_dims(other)
        return MPS([
            np.einsum("iar,jas->ijars", a, b).reshape(
                a.shape[0]*b.shape[0], a.shape[1], a.shape[2]*b.shape[2])
            for a, b in zip(self.cores, other.cores)
        ])

    def right_canonical(self):
        """Leave the norm and Schmidt weights at the FIRST site."""
        cores = [c.copy() for c in self.cores]
        for j in range(len(cores)-1, 0, -1):
            l, d, r = cores[j].shape
            q, rr = _qr(cores[j].reshape(l, d*r).T)
            cores[j] = q.T.reshape(q.shape[1], d, r)
            cores[j-1] = np.einsum("lar,rs->las", cores[j-1], rr.T)
        return MPS(cores)

    def round(self, truncation=Truncation()):
        """Right QR, then left SVD; reports local discarded norms, not a
        propagated physical-error bound for a sequence of HEOM steps.
        """
        cores = self.right_canonical().cores
        errors = []
        for j in range(len(cores)-1):
            l, d, r = cores[j].shape
            u, s, vh, err = split(cores[j].reshape(l*d, r), truncation)
            cores[j] = u.reshape(l, d, len(s))
            cores[j+1] = np.einsum("ij,jar->iar", s[:, None]*vh, cores[j+1])
            errors.append(err)
        return MPS(cores), errors

    def save(self, path):
        """Exclusive creation: never silently overwrite a saved run."""
        with open(path, "xb") as stream:
            np.savez_compressed(stream, **{f"core_{i}": c for i, c in enumerate(self.cores)})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls([data[f"core_{j}"] for j in range(len(data.files))])


def linear_combination(states, weights, truncation=Truncation()):
    if not states or len(states) != len(weights):
        raise ValueError("states and weights must have equal nonzero length")
    out, errors = states[0].scaled(weights[0]), []
    for state, weight in zip(states[1:], weights[1:]):
        out, err = out.add(state.scaled(weight)).round(truncation)
        errors.extend(err)
    return out, errors
