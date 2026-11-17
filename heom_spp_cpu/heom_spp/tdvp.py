"""Functional CPU KSL/TDVP for general complex HEOM generators.

Projector-splitting sweep adapted from the HEOM-MPS implementations by
Qiang Shi, Nan Sheng and Yaming Yan (JCP 148, 174102, 2018), including the
local HEOM_two-qubit-gate and QEC_paper/cpu_heom codes. No global params,
CuPy, state normalization, or Hermitian-generator assumption is used.
"""
import numpy as np
from math import prod
from scipy.linalg import expm

_PATHS = {}


def contract(signature, *arrays):
    key = (signature, tuple(a.shape for a in arrays))
    if key not in _PATHS:
        _PATHS[key] = np.einsum_path(signature, *arrays, optimize="greedy")[0]
    return np.einsum(signature, *arrays, optimize=_PATHS[key])


def arnoldi_action(action, vector, dt, *, dimension=24, tolerance=1e-11):
    """Apply exp(dt*A) with general Arnoldi and double reorthogonalization.

    Zero inputs and happy breakdown are explicit. The residual estimate is
    local, not a global error bound. Budget exhaustion raises rather than
    silently claiming convergence; increase dimension or reduce the step.
    """
    shape = vector.shape
    flat = np.asarray(vector, complex).reshape(-1)
    norm = np.linalg.norm(flat)
    if norm == 0 or dt == 0:
        return vector.copy(), 0
    size = min(dimension, flat.size)
    if size < 1 or tolerance <= 0:
        raise ValueError("positive Arnoldi dimension and tolerance required")
    basis = np.empty((size, flat.size), complex)
    h = np.zeros((size+1, size), complex)
    basis[0] = flat/norm
    for j in range(size):
        work = np.asarray(action(basis[j].reshape(shape))).reshape(-1).copy()
        action_norm = np.linalg.norm(work)
        for _ in range(2):
            projection = basis[:j+1].conj() @ work
            h[:j+1, j] += projection
            work -= projection @ basis[:j+1]
        beta = np.linalg.norm(work)
        h[j+1, j] = beta
        breakdown = beta <= 32*np.finfo(float).eps*max(action_norm, 1e-300)
        if breakdown or j == size-1 or (j >= 3 and (j+1) % 4 == 0):
            coefficients = expm(dt*h[:j+1, :j+1])[:, 0]
            result = norm*(coefficients @ basis[:j+1])
            estimate = abs(dt)*beta*abs(coefficients[-1])*norm
            if breakdown or estimate <= tolerance*max(norm, np.linalg.norm(result)):
                if not np.all(np.isfinite(result)):
                    raise FloatingPointError("nonfinite Arnoldi exponential")
                return result.reshape(shape), j+1
        if j+1 < size:
            basis[j+1] = work/beta
    raise RuntimeError(
        f"Arnoldi budget {size} exhausted (estimated local residual {estimate:.3g}); "
        "increase --krylov-dim or decrease --max-dt")


def pack(operator):
    """MPO site (left, out, in, right) reshaped once for the GEMM kernels."""
    w, o, i, z = operator.shape
    return (np.ascontiguousarray(operator.transpose(0, 2, 1, 3)).reshape(w*i, o*z),
            np.ascontiguousarray(operator.transpose(2, 3, 0, 1)).reshape(i*z, w*o),
            operator.shape)


# The four contractions below are the einsums
#   K      awb,woiz,bic,dzc->aod      S      -awb,bc,dwc->ad
#   left   awb,aod,woiz,bic->dzc      right  aod,woiz,bic,dzc->awb
# written as three (two) matrix products, following the compiled kernels of
# the independent_heom package: at the small ranks used here einsum/tensordot
# bookkeeping, not BLAS, dominates the cost.
def _pack_right_k(right):
    d, z, c = right.shape
    return np.ascontiguousarray(right.transpose(2, 1, 0)).reshape(c*z, d)


def _apply_k(x, left, packed, right_k):
    a, w, b = left.shape
    _, i, c = x.shape
    wk, _, (_, o, _, z) = packed
    t = (left.reshape(a*w, b)@x.reshape(b, i*c)).reshape(a, w, i, c)
    t = (t.transpose(0, 3, 1, 2).reshape(a*c, w*i)@wk).reshape(a, c, o, z)
    return (t.transpose(0, 2, 1, 3).reshape(a*o, c*z)@right_k).reshape(a, o, -1)


def _pack_right_s(right):
    d, w, c = right.shape
    return np.ascontiguousarray(right.transpose(1, 2, 0)).reshape(w*c, d)


def _apply_s(x, left, right_s):
    a, w, b = left.shape
    return -((left.reshape(a*w, b)@x).reshape(a, -1)@right_s)


def _left(environment, packed, core):
    a, w, b = environment.shape
    _, i, c = core.shape
    wk, _, (_, o, _, z) = packed
    t = (environment.reshape(a*w, b)@core.reshape(b, i*c)).reshape(a, w, i, c)
    t = (t.transpose(0, 3, 1, 2).reshape(a*c, w*i)@wk).reshape(a, c, o, z)
    return (core.conj().reshape(a*o, -1).T
            @ t.transpose(0, 2, 3, 1).reshape(a*o, z*c)).reshape(-1, z, c)


def _right(environment, packed, core):
    d, z, c = environment.shape
    b, i, _ = core.shape
    _, wr, (w, o, _, _) = packed
    t = (core.reshape(b*i, c)@environment.transpose(2, 0, 1).reshape(c, d*z)).reshape(b, i, d, z)
    t = (t.transpose(0, 2, 1, 3).reshape(b*d, i*z)@wr).reshape(b, d, w, o)
    a = core.shape[0]
    return (core.conj().reshape(a, o*d)
            @ t.transpose(3, 1, 2, 0).reshape(o*d, w*b)).reshape(a, w, b)


def _qr_left(core):
    left, physical, right = core.shape
    q, r = np.linalg.qr(core.reshape(left*physical, right), mode="reduced")
    return q.reshape(left, physical, q.shape[1]), r


def _qr_right(core):
    left, physical, right = core.shape
    q, r = np.linalg.qr(core.reshape(left, physical*right).conj().T, mode="reduced")
    return r.conj().T, q.conj().T.reshape(q.shape[1], physical, right)


def expand_bonds(cores, maximum):
    """Exact padding after an RK4 warm start introduced correlated directions.

    One-site TDVP uses a fixed-rank manifold. Padding is not an adaptive-rank
    algorithm; check convergence with the chosen maximum.
    """
    dims = [a.shape[1] for a in cores]
    bonds = [1]+[min(maximum, prod(dims[:i]), prod(dims[i:]))
                 for i in range(1, len(dims))]+[1]
    out = []
    for i, a in enumerate(cores):
        if a.shape[0] > bonds[i] or a.shape[2] > bonds[i+1]:
            raise ValueError("input rank exceeds TDVP rank; round the warm start first")
        b = np.zeros((bonds[i], dims[i], bonds[i+1]), complex)
        b[:a.shape[0], :, :a.shape[2]] = a
        out.append(b)
    return out


def ksl_step(cores, operators, dt, *, krylov_dim=24, krylov_tol=1e-11,
             system_channels=None, packed=None):
    """Symmetric one-site projector splitting; general non-Hermitian MPO.

    MPO axes: (left, output, input, right). MPS cores are copied so original
    saved HEOM states and shared incoming memory are never changed. packed is
    [pack(w) for w in operators], cached by the caller across steps.
    """
    cores = [np.ascontiguousarray(a, dtype=complex).copy() for a in cores]
    count = len(cores)
    if count != len(operators):
        raise ValueError("MPS and MPO site counts differ")
    if packed is None:
        packed = [pack(w) for w in operators]
    boundary = np.ones((1, 1, 1), complex)
    budgets = []

    def local(a, left, right, time, site):
        if site == 0 and system_channels is not None:
            def action(x):
                result = np.zeros_like(x)
                for j, matrix in enumerate(system_channels):
                    if matrix is not None:
                        result[0] += matrix.dot(x[0]) @ right[:, j, :].T
                return left[0, 0, 0]*result
        else:
            right_k = _pack_right_k(right)
            left = np.ascontiguousarray(left)

            def action(x):
                return _apply_k(x, left, packed[site], right_k)
        result, used = arnoldi_action(
            action, a, time, dimension=krylov_dim, tolerance=krylov_tol)
        budgets.append(used)
        return result

    def bond(a, left, right, time):
        right_s = _pack_right_s(right)
        left = np.ascontiguousarray(left)
        result, used = arnoldi_action(
            lambda x: _apply_s(x, left, right_s),
            a, time, dimension=krylov_dim, tolerance=krylov_tol)
        budgets.append(used)
        return result

    right_env = [None]*(count+1)
    right_env[-1] = boundary
    for i in range(count-1, 0, -1):
        r, q = _qr_right(cores[i])
        cores[i] = q
        cores[i-1] = np.tensordot(cores[i-1], r, axes=(2, 0))
        right_env[i] = _right(right_env[i+1], packed[i], q)
    left_env = [boundary]
    for i in range(count-1):
        a = local(cores[i], left_env[i], right_env[i+1], dt/2, i)
        q, r = _qr_left(a)
        cores[i] = q
        if i == 0 and system_channels is not None:
            a = q[0]
            environment = np.stack([
                np.zeros((a.shape[1], a.shape[1]), complex) if matrix is None
                else a.conj().T @ matrix.dot(a) for matrix in system_channels], axis=1)
            left_env.append(left_env[i][0, 0, 0]*environment)
        else:
            left_env.append(_left(left_env[i], packed[i], q))
        r = bond(r, left_env[i+1], right_env[i+1], dt/2)
        cores[i+1] = np.tensordot(r, cores[i+1], axes=(1, 0))
    cores[-1] = local(cores[-1], left_env[-1], boundary, dt, count-1)
    right = boundary
    for i in range(count-1, 0, -1):
        r, q = _qr_right(cores[i])
        cores[i] = q
        right = _right(right, packed[i], q)
        r = bond(r, left_env[i], right, dt/2)
        cores[i-1] = np.tensordot(cores[i-1], r, axes=(2, 0))
        cores[i-1] = local(cores[i-1], left_env[i-1], right, dt/2, i-1)
    if any(not np.all(np.isfinite(a)) for a in cores):
        raise FloatingPointError("nonfinite TDVP state")
    return cores, dict(local_exponentials=len(budgets), max_krylov_used=max(budgets))
