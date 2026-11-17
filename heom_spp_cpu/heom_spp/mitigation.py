"""History convolution, PEA, and alternating least-squares PEC inversion."""
from math import prod

import numpy as np

from itertools import product

from .mps import MPS, Truncation
from .pauli import paulis, walsh


def amplify(p, n, gain, truncation=Truncation()):
    if not isinstance(gain, int) or gain < 0:
        raise ValueError("gain must be a nonnegative integer")
    f = walsh(p, n)
    out = MPS.product([np.ones(d) for d in f.dims])
    errors = []
    for _ in range(gain):
        out, err = out.hadamard(f).round(truncation)
        errors.extend(err)
    return walsh(out, n, inverse=True), dict(discarded_norm_sum=sum(errors), gain=gain)


def inverse_dense(p, n, *, max_entries=1_000_000, min_abs=1e-10):
    f = walsh(p, n).dense(max_entries)
    if np.min(np.abs(f)) <= min_abs:
        raise ValueError("singular/ill-conditioned Walsh inverse")
    return walsh(MPS.from_dense(1/f), n, inverse=True)


def real_tensor(p):
    """Real part via real block cores; no dense enumeration."""
    if len(p.cores) == 1:
        return MPS([p.cores[0].real])
    out = [np.concatenate((p.cores[0].real, p.cores[0].imag), axis=2)]
    for c in p.cores[1:-1]:
        out.append(np.concatenate((np.concatenate((c.real, c.imag), axis=2),
                                   np.concatenate((-c.imag, c.real), axis=2)), axis=0))
    out.append(np.concatenate((p.cores[-1].real, -p.cores[-1].imag), axis=0))
    return MPS(out).round(Truncation(cutoff=1e-13))[0]


def _absorb(tensor, core):
    """tensor (label, x, rest...), core (x, label, y) -> (label, rest..., y).

    One batched matrix product per call: the label is shared, never summed.
    """
    shape = tensor.shape
    moved = np.moveaxis(tensor, 1, -1).reshape(shape[0], -1, shape[1])
    return (moved@core.transpose(1, 0, 2)).reshape(shape[0], *shape[2:], core.shape[2])


def _push(quadratic, linear, c, g):
    """Extend the environments by one site.

    quadratic (q, q, F, F) is the environment of |F_p F_q|^2 with the two
    copies of F kept as separate legs: F*F is never formed, so memory is
    bond(q)^2 bond(F)^2 instead of bond(F)^4 per label. linear is (q, F).
    Right environments use the same routine with transposed cores.
    """
    t = np.moveaxis(np.tensordot(quadratic, c, axes=(0, 0)), 3, 0)  # (label, q, F, F, q')
    quadratic = _absorb(_absorb(_absorb(t, c), g), g).sum(axis=0)
    linear = np.einsum("ij,iar,jas->rs", linear, c, g, optimize=True)
    return quadratic, linear


def _flip(core):
    return core.transpose(2, 1, 0)


def _center(cores, site):
    """Move the orthogonality center without truncating the inverse ansatz."""
    for j in range(site):
        l, d, r = cores[j].shape
        u, v = np.linalg.qr(cores[j].reshape(l*d, r), mode="reduced")
        cores[j] = u.reshape(l, d, -1)
        cores[j+1] = np.einsum("ij,jar->iar", v, cores[j+1])
    for j in range(len(cores)-1, site, -1):
        l, d, r = cores[j].shape
        u, v = np.linalg.qr(cores[j].reshape(l, d*r).T, mode="reduced")
        cores[j] = u.T.reshape(-1, d, r)
        cores[j-1] = np.einsum("lar,rs->las", cores[j-1], v.T)


def inverse_als(p, n, *, bond=4, sweeps=12, ridge=1e-12, tolerance=1e-10,
                seed=7, max_local_unknowns=4096, progress=None):
    """Minimize ||F_p F_q - 1||² with F_q(0,...,0)=1.

    Returns signed q and convergence history. This is an alternating local
    optimizer, not a guarantee of global convergence/invertibility, and does
    not minimize the quasiprobability norm Gamma. ridge penalizes local tensor
    norm in the current canonical gauge, not Gamma. Full residual is divided
    by the number of histories for reporting.
    """
    if bond < 1 or sweeps < 1 or ridge < 0 or tolerance < 0:
        raise ValueError("invalid inverse optimizer parameters")
    f_original = walsh(p, n)
    f = real_tensor(f_original)
    # Verify reality as a tensor; local cores can carry arbitrary complex gauges.
    imaginary = f_original.add(f.scaled(-1))
    if imaginary.norm() > 1e-10 * max(1, f.norm()):
        raise ValueError("Walsh tensor must be real")
    dims, length = f.dims, len(f.cores)
    rng = np.random.default_rng(seed)
    ranks = [1] + [min(bond, prod(dims[:j]), prod(dims[j:])) for j in range(1, length)] + [1]
    # Start from the layer-wise inverse 1/f_j(b), f_j the Walsh transform of
    # the marginal of layer j, so that any bond does at least as well as
    # independent PEC; small noise opens the extra bond directions.
    lefts, rights = [np.ones(1)], [np.ones(1)]
    for c in f.cores[:-1]:
        lefts.append(lefts[-1]@c[:, 0, :])
    for c in reversed(f.cores[1:]):
        rights.insert(0, c[:, 0, :]@rights[0])
    cores = []
    for j, dim in enumerate(dims):
        marginal = np.einsum("l,lar,r->a", lefts[j], f.cores[j], rights[j]).real
        c = rng.normal(scale=1e-3, size=(ranks[j], dim, ranks[j+1]))
        c[0, :, 0] += np.where(np.abs(marginal) > 1e-6, 1/np.where(marginal == 0, 1, marginal), 1.)
        cores.append(c)
    history, total, stalled = [], float(prod(dims)), None
    g = [c.real for c in f.cores]
    # Environments are cached and moved one site at a time together with the
    # orthogonality center: O(L) contractions per sweep.
    _center(cores, 0)
    empty = (np.ones((1, 1, 1, 1)), np.ones((1, 1)))
    lefts_env, rights_env = [empty]+[None]*(length-1), [None]*(length-1)+[empty]
    for j in range(length-1, 0, -1):
        rights_env[j-1] = _push(*rights_env[j], _flip(cores[j]), _flip(g[j]))
    previous = 0
    for sweep in range(sweeps):
        for site in [*range(length), *range(length-2, -1, -1)]:
            if site == previous+1:
                l, d, r = cores[previous].shape
                u, v = np.linalg.qr(cores[previous].reshape(l*d, r), mode="reduced")
                cores[previous] = u.reshape(l, d, -1)
                cores[site] = np.einsum("ij,jar->iar", v, cores[site])
                lefts_env[site] = _push(*lefts_env[previous], cores[previous], g[previous])
            elif site == previous-1:
                l, d, r = cores[previous].shape
                u, v = np.linalg.qr(cores[previous].reshape(l, d*r).T, mode="reduced")
                cores[previous] = u.T.reshape(-1, d, r)
                cores[site] = np.einsum("lar,rs->las", cores[site], v.T)
                rights_env[site] = _push(*rights_env[previous], _flip(cores[previous]),
                                         _flip(g[previous]))
            previous = site
            (left, lin_left), (right, lin_right) = lefts_env[site], rights_env[site]
            l, d, r = cores[site].shape
            # matrices[label] = sum left[i,j,k,m] F[k,label,t] F[m,label,u] right[r,s,t,u]
            t = np.moveaxis(np.tensordot(left, g[site], axes=(2, 0)), 3, 0)  # (label, i, j, m, t)
            t = _absorb(np.moveaxis(t, 3, 1), g[site])                          # (label, i, j, t, u)
            matrices = np.tensordot(t, right, axes=([3, 4], [2, 3])).transpose(0, 1, 3, 2, 4)
            matrices = matrices.reshape(d, l*r, l*r)
            rhs_all = np.einsum("ij,jat,rt->air", lin_left, g[site], lin_right,
                                optimize=True).reshape(d, l*r)
            if l*r > max_local_unknowns:
                raise ValueError("local dense normal equation exceeds max_local_unknowns")
            left0, right0 = np.ones(1), np.ones(1)
            for c in cores[:site]:
                left0 = left0 @ c[:, 0, :]
            for c in reversed(cores[site+1:]):
                right0 = c[:, 0, :] @ right0
            constraint = np.outer(left0, right0).reshape(-1)
            for label in range(d):
                matrix = (matrices[label]+matrices[label].T)/2 + ridge*np.eye(l*r)
                rhs = rhs_all[label]
                if label == 0:
                    kkt = np.zeros((l*r+1, l*r+1))
                    kkt[:-1, :-1] = matrix
                    kkt[:-1, -1] = kkt[-1, :-1] = constraint
                    sol = np.linalg.lstsq(kkt, np.r_[rhs, 1.0], rcond=1e-14)[0][:-1]
                else:
                    sol = np.linalg.lstsq(matrix, rhs, rcond=1e-14)[0]
                cores[site][:, label, :] = sol.reshape(l, r)
        fq = MPS(cores)
        # The sweep ends at site 0, whose local problem is the whole objective:
        # ||F_p F_q - 1||^2 = x.M.x - 2 rhs.x + N. Free, but the three terms
        # cancel, so it cannot resolve an rms residual below about 1e-7. The
        # residual tensor itself is formed (most of the run time for a large
        # bond) only in that regime and after the last sweep.
        x = cores[0].transpose(1, 0, 2).reshape(dims[0], -1)
        loss = max(float(np.einsum("ai,aij,aj->", x, matrices, x)-2*np.sum(rhs_all*x)+total)/total, 0.)
        exact = sweep == sweeps-1 or loss <= max(4*tolerance**2, 1e-13)
        if exact:
            residual = f.hadamard(fq).add(MPS.product([np.ones(d) for d in dims]).scaled(-1))
            loss = residual.norm()**2/total
        norm_error = float(abs(fq.value([0]*length)-1))
        history.append(dict(sweep=sweep+1, residual_mse=loss, normalization_error=norm_error,
                            exact_residual=bool(exact)))
        if progress:
            progress(history[-1])
        if loss <= tolerance**2 and norm_error <= tolerance:
            break
        if exact and stalled is not None and loss > .99*stalled:
            break  # below the cheap estimate's resolution and no longer improving
        stalled = loss if exact else None
    q = walsh(fq, n, inverse=True)
    return q, dict(converged=history[-1]["residual_mse"] <= tolerance**2
                   and history[-1]["normalization_error"] <= tolerance,
                   history=history, bonds=list(q.bonds), ridge=ridge,
                   objective="mean squared Walsh convolution residual")


def quasiprobability_report(q, *, max_entries=1_000_000):
    dense = q.dense(max_entries)
    if np.max(np.abs(dense.imag)) > 1e-9:
        raise ValueError("quasiprobability has a significant imaginary part")
    gamma = float(np.abs(dense.real).sum())
    return dict(gamma=gamma, gamma_squared=gamma**2,
                normalization_error=float(abs(dense.sum()-1)),
                method="exhaustive signed-history enumeration")


def sample_corrections(q, shots, rng, *, max_entries=1_000_000):
    if not isinstance(shots, int) or shots < 0:
        raise ValueError("shots must be a nonnegative integer")
    report = quasiprobability_report(q, max_entries=max_entries)
    values = q.dense(max_entries).real.reshape(-1)
    gamma = report["gamma"]
    if gamma == 0:
        raise ValueError("zero correction tensor")
    indices = rng.choice(len(values), size=shots, p=np.abs(values)/gamma)
    histories = np.array(np.unravel_index(indices, q.dims)).T
    return histories, gamma*np.sign(values[indices]), report


def history_expectation(slots, history, observable, initial=None):
    """<O> after the ideal circuit with Pauli error P_a inserted AFTER each U_j.

    Dense state vector on the small register; ideal layers come from
    slot.ideal. This does not model noisy corrections or twirl pulses.
    """
    if not slots or len(history) != len(slots):
        raise ValueError("one label per slot required")
    d = len(slots[0].ideal)
    n = d.bit_length()-1
    basis = paulis(n)
    if any(not isinstance(a, (int, np.integer)) or not 0 <= a < 4**n for a in history):
        raise ValueError("invalid Pauli history")
    if initial is None:
        state = np.eye(d, dtype=complex)[0]
    else:
        state = np.asarray(initial, complex)
        if state.shape != (d,) or not np.isclose(np.linalg.norm(state), 1):
            raise ValueError("initial pure state must be normalized")
    observable = np.asarray(observable, complex)
    if observable.shape != (d, d) or not np.allclose(observable, observable.conj().T):
        raise ValueError("observable must be Hermitian")
    for slot, label in zip(slots, history):
        state = basis[label]@(slot.ideal@state)
    return float(np.real(state.conj()@observable@state))


def response_tensor(slots, observable, initial=None, max_entries=65536):
    n = len(slots[0].ideal).bit_length()-1
    m = 4**n
    if m**len(slots) > max_entries:
        raise ValueError("response enumeration exceeds max_entries; use sampled histories")
    return np.array([history_expectation(slots, h, observable, initial)
                     for h in product(range(m), repeat=len(slots))]).reshape((m,)*len(slots))
