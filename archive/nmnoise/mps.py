"""Actual spatial MPS trajectory propagation using local gates and SVD.

This compresses the SYSTEM state, not the temporal influence functional or
HEOM hierarchy. Memory is supplied by the same colored-noise paths as dense.
Without a bond cap/cutoff there is no SVD truncation; Strang time splitting
remains and is checked against the unsplit dense exponentials.
"""
import numpy as np
from scipy.linalg import expm
from .core import ket0, timeline


class MPS:
    def __init__(self, tensors, max_bond=None, svd_cutoff=0.0):
        self.tensors = tensors
        self.max_bond = max_bond
        self.svd_cutoff = svd_cutoff
        self.discarded_weight_sum = 0.0
        self.peak_bond = max(a.shape[-1] for a in tensors)

    @classmethod
    def from_state(cls, psi, n, max_bond=None, svd_cutoff=0.0):
        if max_bond is not None and max_bond < 1:
            raise ValueError("max_bond must be positive")
        if svd_cutoff < 0:
            raise ValueError("svd_cutoff must be nonnegative")
        # Exact initial factorization; truncation is applied during evolution.
        tensors, left, work = [], 1, np.asarray(psi).reshape(1, -1)
        for q in range(n-1):
            u, s, vh = np.linalg.svd(work.reshape(left*2, -1), full_matrices=False)
            keep = max(1, int(np.count_nonzero(s > 1e-14)))
            tensors.append(u[:, :keep].reshape(left, 2, keep))
            work = s[:keep, None]*vh[:keep]
            left = keep
        tensors.append(work.reshape(left, 2, 1))
        return cls(tensors, max_bond, svd_cutoff)

    def one(self, gate, q):
        self.tensors[q] = np.einsum("ij,ajb->aib", gate, self.tensors[q])

    def canonicalize_pair(self, q):
        # Put orthogonality centre on pair (q,q+1), so discarded singular-value
        # weight is physically meaningful even when gates move along the chain.
        for k in range(q):
            a = self.tensors[k]
            u, r = np.linalg.qr(a.reshape(-1, a.shape[-1]))
            self.tensors[k] = u.reshape(a.shape[0], 2, -1)
            self.tensors[k+1] = np.einsum("ab,bic->aic", r, self.tensors[k+1])
        for k in range(len(self.tensors)-1, q+1, -1):
            a = self.tensors[k]
            u, r = np.linalg.qr(a.reshape(a.shape[0], -1).T)
            self.tensors[k] = u.T.reshape(-1, 2, a.shape[-1])
            self.tensors[k-1] = np.einsum("aib,bc->aic", self.tensors[k-1], r.T)

    def two(self, gate, q):
        self.canonicalize_pair(q)
        a, b = self.tensors[q:q+2]
        theta = np.einsum("aib,bjc->aijc", a, b)
        theta = np.einsum("IJij,aijc->aIJc", gate.reshape(2,2,2,2), theta)
        u, s, vh = np.linalg.svd(theta.reshape(a.shape[0]*2, 2*b.shape[-1]), full_matrices=False)
        keep = max(1, int(np.count_nonzero(s > self.svd_cutoff)))
        if self.max_bond is not None:
            keep = min(keep, self.max_bond)
        self.discarded_weight_sum += float(np.sum(s[keep:]**2))
        self.tensors[q] = u[:, :keep].reshape(a.shape[0], 2, keep)
        self.tensors[q+1] = (s[:keep, None]*vh[:keep]).reshape(keep, 2, b.shape[-1])
        self.peak_bond = max(self.peak_bond, keep)

    def state(self):
        a = self.tensors[0]
        for b in self.tensors[1:]:
            a = np.tensordot(a, b, axes=(-1, 0))
        return a.reshape(-1)


def mps_trajectories(pulses, noise, dt, initial=None, substeps=1,
                     max_bond=None, svd_cutoff=0.0):
    samples, steps, n = noise.shape
    ids = timeline(pulses, dt)
    if steps != len(ids) or substeps < 1 or int(substeps) != substeps:
        raise ValueError("Invalid trajectory length or substeps")
    psi0 = ket0(n) if initial is None else np.asarray(initial, complex)
    h = dt/substeps
    controls = [expm(-1j*h*p.h) for p in pulses]
    states = np.empty((samples, 2**n), complex)
    peak_bond, discarded, norm_error = 1, 0.0, 0.0
    for b in range(samples):
        mps = MPS.from_state(psi0, n, max_bond, svd_cutoff)
        for t, k in enumerate(ids):
            p = pulses[k]
            # exp(-i*(xi Z/2)*(h/2)) on either side of the control.
            phases = np.exp(-0.25j*h*noise[b,t])
            for _ in range(substeps):
                for q in range(n):
                    mps.one(np.diag([phases[q], phases[q].conjugate()]), q)
                if len(p.sites) == 1:
                    mps.one(controls[k], p.sites[0])
                else:
                    mps.two(controls[k], p.sites[0])
                for q in range(n):
                    mps.one(np.diag([phases[q], phases[q].conjugate()]), q)
        psi = mps.state()
        norm = np.linalg.norm(psi)
        if norm < 1e-12:
            raise RuntimeError("MPS truncation destroyed the state")
        states[b] = psi/norm
        norm_error = max(norm_error, abs(norm-1))
        discarded = max(discarded, mps.discarded_weight_sum)
        peak_bond = max(peak_bond, mps.peak_bond)
    return states, dict(peak_bond=peak_bond,
                       max_trajectory_discarded_weight_sum=discarded,
                       max_trajectory_norm_error_before_normalization=norm_error)
