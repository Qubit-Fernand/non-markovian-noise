"""Exact SPP for static classical Gaussian noise, by quadrature instead of HEOM.

A static mode k contributes delta_k * gain * Q_k to the Hamiltonian, with
delta_k ~ N(0, gwval_k) drawn once and kept for the whole circuit. For a fixed
delta the evolution is unitary, so the twirled Pauli errors of different layers
are independent and

    p(a_1..a_L) = integral d delta  P(delta)  prod_j p_j(a_j | delta),
    p_j(a | delta) = |Tr(P_a W_j(delta) U_j^dagger) / d|^2,

with W_j the noisy and U_j the ideal layer unitary (gate-after convention, as
in spp.build_spp). The integral is a tensor-product Gauss-Hermite rule; every
node is one product term of the time MPS, which is then compressed.

This has no hierarchy depth, HEOM bond or time step. Its only numerical
parameter is the number of nodes per mode, checked by `static_reference_check`.
It is the yardstick for the HEOM parameters, and only exists for static noise.
"""
from math import prod
from time import perf_counter

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

from .mps import MPS, Truncation, split
from .pauli import paulis
from .pulse import Frame, Segment


def static_nodes(modes, nodes, max_nodes=300_000):
    """Tensor-product Gauss-Hermite rule: delta (N, modes) and weights (N,)."""
    if nodes < 1:
        raise ValueError("at least one quadrature node per mode required")
    x, w = hermegauss(nodes)
    w = w/w.sum()
    axes, weights = [], []
    for mode in modes:
        if mode.action != "commutator" or mode.wval != 0:
            raise ValueError("the quadrature reference needs static classical modes (wval = 0)")
        if mode.sigma == 0:
            axes.append(np.zeros(1))
            weights.append(np.ones(1))
        else:
            axes.append(mode.sigma*x)
            weights.append(w)
    if prod(len(a) for a in axes) > max_nodes:
        raise ValueError("too many quadrature nodes; lower nodes or raise max_nodes")
    delta = np.stack([g.reshape(-1) for g in np.meshgrid(*axes, indexing="ij")], axis=1)
    weight = np.ones(1)
    for v in weights:
        weight = np.multiply.outer(weight, v).reshape(-1)
    return delta, weight


def layer_probabilities(engine, slot, delta, basis, chunk=2048):
    """p_j(a | delta) for every node: array (N, 4**n)."""
    d = engine.d
    operators = np.array([m.operator for m in engine.modes])
    out = np.empty((len(delta), len(basis)))
    for start in range(0, len(delta), chunk):
        x = delta[start:start+chunk]
        w = np.broadcast_to(np.eye(d, dtype=complex), (len(x), d, d))
        for operation in slot.segments:
            if isinstance(operation, Frame):
                w = operation.unitary@w
                continue
            if not isinstance(operation, Segment):
                raise ValueError("the quadrature reference does not support bath resets")
            gains = np.array([m.gain(operation.gains) for m in engine.modes])
            h = operation.hamiltonian+engine.drift+np.einsum("nk,kij->nij", x*gains, operators)
            energy, vectors = np.linalg.eigh(h)
            phase = np.exp(-1j*operation.duration*energy)
            w = (vectors*phase[:, None, :])@vectors.conj().transpose(0, 2, 1)@w
        error = w@slot.ideal.conj().T
        out[start:start+chunk] = np.abs(np.einsum("aij,nji->na", basis, error)/d)**2
    return out


def mixture_mps(tables, weights, truncation=Truncation(64, 1e-13)):
    """MPS of sum_n weights[n] prod_j tables[j][n, a_j], compressed right to left."""
    if len(tables) == 1:
        return MPS([(weights@tables[0]).reshape(1, -1, 1)]), 0.
    # The weights ride in the carry, so far-out nodes of negligible weight do
    # not cost bond dimension. Rows of tables[0] have norm <= 1, hence the
    # error of the result is at most sqrt(N) times the discarded norm.
    cores, carry, discarded = [], np.asarray(weights, float)[:, None], 0.
    for table in reversed(tables[1:]):
        joint = (table[:, :, None]*carry[:, None, :]).reshape(len(weights), -1)
        u, s, vh, error = split(joint, truncation)
        cores.append(vh.reshape(len(s), table.shape[1], carry.shape[1]))
        carry = u*s
        discarded += error
    first = np.einsum("na,nr->ar", tables[0], carry)
    p, errors = MPS([first[None], *reversed(cores)]).round(truncation)
    return p, discarded+sum(errors)


def static_reference(engine, slots, *, nodes=6, truncation=Truncation(64, 1e-13)):
    """Exact SPP of static classical noise on a closed system. Returns (MPS, info)."""
    if engine.whites or np.any(engine.markov):
        raise ValueError("the quadrature reference needs a closed system (no T1, no white noise)")
    start = perf_counter()
    delta, weights = static_nodes(engine.modes, nodes)
    basis = paulis(engine.n)
    tables = [layer_probabilities(engine, slot, delta, basis) for slot in slots]
    p, discarded = mixture_mps(tables, weights, truncation)
    return p, dict(nodes_per_mode=nodes, quadrature_nodes=len(weights),
        compression_discarded_norm_sum=discarded, bonds=list(p.bonds),
        unitarity_error=float(max(np.max(np.abs(t.sum(axis=1)-1)) for t in tables)),
        seconds=perf_counter()-start)


def static_reference_check(engine, slots, *, nodes=6, truncation=Truncation(64, 1e-13)):
    """Reference at `nodes`, plus its distance to the rule with two nodes fewer.

    That distance is essentially the error of the COARSER rule, so it is a
    conservative estimate for the returned one.
    """
    p, info = static_reference(engine, slots, nodes=nodes, truncation=truncation)
    if nodes > 2:
        coarse, _ = static_reference(engine, slots, nodes=nodes-2, truncation=truncation)
        info["quadrature_check_nodes"] = nodes-2
        info["quadrature_check_l2"] = p.add(coarse.scaled(-1)).norm()
    return p, info
