"""History convolution, PEA, and alternating least-squares PEC inversion."""
from math import prod

import numpy as np

from .mps import K, MPS, Truncation
from .pauli import walsh


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


def _environments(q, f2, f, site):
    # Quadratic environment (q bra bond, q ket bond, F^2 bond);
    # linear environment (q bond, F bond). Recomputed for clarity, O(L^2)
    # contractions per sweep; no history-space array is allocated.
    a, b = np.ones((1, 1, 1)), np.ones((1, 1))
    for j in range(site):
        c, w, g = q[j], f2[j], f[j]
        a = K.einsum("ijk,iar,jas,kat->rst", a, c, c, w)
        b = K.einsum("ij,iar,jas->rs", b, c, g)
    ar, br = np.ones((1, 1, 1)), np.ones((1, 1))
    for j in range(len(q)-1, site, -1):
        c, w, g = q[j], f2[j], f[j]
        ar = K.einsum("rst,iar,jas,kat->ijk", ar, c, c, w)
        br = K.einsum("rs,iar,jas->ij", br, c, g)
    return a, ar, b, br


def _center(cores, site):
    """Move the orthogonality center without truncating the inverse ansatz."""
    for j in range(site):
        l, d, r = cores[j].shape
        u, v = K.qr(cores[j].reshape(l*d, r), pivot_axis=1)
        cores[j] = u.reshape(l, d, -1)
        cores[j+1] = K.einsum("ij,jar->iar", v, cores[j+1])
    for j in range(len(cores)-1, site, -1):
        l, d, r = cores[j].shape
        u, v = K.qr(cores[j].reshape(l, d*r).T, pivot_axis=1)
        cores[j] = u.T.reshape(-1, d, r)
        cores[j-1] = K.einsum("lar,rs->las", cores[j-1], v.T)


def inverse_als(p, n, *, bond=4, sweeps=12, ridge=1e-12, tolerance=1e-10,
                seed=7, max_local_unknowns=4096):
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
    f2 = f.hadamard(f).round(Truncation(cutoff=1e-13))[0]
    dims, length = f.dims, len(f.cores)
    rng = np.random.default_rng(seed)
    ranks = [1] + [min(bond, prod(dims[:j]), prod(dims[j:])) for j in range(1, length)] + [1]
    cores = []
    for j, dim in enumerate(dims):
        c = rng.normal(scale=.03, size=(ranks[j], dim, ranks[j+1]))
        c[0, :, 0] += 1
        cores.append(c)
    history = []
    for sweep in range(sweeps):
        for site in [*range(length), *range(length-2, -1, -1)]:
            _center(cores, site)
            left, right, lin_left, lin_right = _environments(cores, f2.cores, f.cores, site)
            l, d, r = cores[site].shape
            if l*r > max_local_unknowns:
                raise ValueError("local dense normal equation exceeds max_local_unknowns")
            left0, right0 = np.ones(1), np.ones(1)
            for c in cores[:site]:
                left0 = left0 @ c[:, 0, :]
            for c in reversed(cores[site+1:]):
                right0 = c[:, 0, :] @ right0
            constraint = np.outer(left0, right0).reshape(-1)
            for label in range(d):
                matrix = K.einsum("ijk,kt,rst->irjs", left,
                                  f2.cores[site][:, label, :], right).reshape(l*r, l*r).real
                matrix = (matrix+matrix.T)/2 + ridge*np.eye(l*r)
                rhs = K.einsum("ij,jt,rt->ir", lin_left,
                               f.cores[site][:, label, :], lin_right).reshape(-1).real
                if label == 0:
                    kkt = np.zeros((l*r+1, l*r+1))
                    kkt[:-1, :-1] = matrix
                    kkt[:-1, -1] = kkt[-1, :-1] = constraint
                    sol = np.linalg.lstsq(kkt, np.r_[rhs, 1.0], rcond=1e-14)[0][:-1]
                else:
                    sol = np.linalg.lstsq(matrix, rhs, rcond=1e-14)[0]
                cores[site][:, label, :] = sol.reshape(l, r)
        fq = MPS(cores)
        residual = f.hadamard(fq).add(MPS.product([np.ones(d) for d in dims]).scaled(-1))
        loss = residual.norm()**2/prod(dims)
        norm_error = float(abs(fq.value([0]*length)-1))
        history.append(dict(sweep=sweep+1, residual_mse=loss, normalization_error=norm_error))
        if loss <= tolerance**2 and norm_error <= tolerance:
            break
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
