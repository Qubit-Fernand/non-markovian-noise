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
