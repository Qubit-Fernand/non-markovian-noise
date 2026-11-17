"""Labels are base four in I,X,Y,Z order, qubit 0 most significant.

With this ordering phase-free Pauli multiplication is integer XOR. The
commutation-character matrix is NOT the ordinary bit-parity Hadamard matrix.
"""
from functools import lru_cache
from itertools import product

import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1, -1]).astype(complex)


@lru_cache(maxsize=8)
def paulis(n):
    if not isinstance(n, int) or n < 1 or n > 5:
        raise ValueError("Pauli probe enumeration supports 1..5 qubits")
    matrices = []
    for word in product((I, X, Y, Z), repeat=n):
        matrix = np.ones((1, 1), complex)
        for p in word:
            matrix = np.kron(matrix, p)
        matrices.append(matrix)
    out = np.array(matrices)
    out.flags.writeable = False
    return out


@lru_cache(maxsize=8)
def characters(n):
    paulis(n)  # validate convention / supported range
    single = np.array([[1, 1, 1, 1], [1, 1, -1, -1],
                       [1, -1, 1, -1], [1, -1, -1, 1]], float)
    out = single
    for _ in range(n-1):
        out = np.kron(out, single)
    out.flags.writeable = False
    return out


def walsh(mps, n, inverse=False):
    matrix = characters(n) / (4**n if inverse else 1)
    if any(d != 4**n for d in mps.dims):
        raise ValueError("expected one n-qubit Pauli leg per time slot")
    out = mps
    for j in range(len(mps.cores)):
        out = out.local(j, matrix)
    return out



def embed(operator, sites, n):
    sites = tuple(sites)
    if len(set(sites)) != len(sites) or any(s < 0 or s >= n for s in sites):
        raise ValueError("invalid sites")
    operator = np.asarray(operator)
    if operator.shape != (2**len(sites),)*2:
        raise ValueError("operator size does not match sites")
    result = np.zeros((2**n, 2**n), complex)
    for column in range(2**n):
        bits = [(column >> (n-1-j)) & 1 for j in range(n)]
        local = sum(bits[s] << (len(sites)-1-j) for j, s in enumerate(sites))
        for target in range(2**len(sites)):
            out = bits.copy()
            for j, site in enumerate(sites):
                out[site] = (target >> (len(sites)-1-j)) & 1
            row = sum(b << (n-1-j) for j, b in enumerate(out))
            result[row, column] = operator[target, local]
    return result
