import numpy as np
import scipy.sparse as sps
import scipy as sp


def generateA(n: int, gamma: float) -> sps.csr_matrix:
    # n: points in the grid
    # gamma: parameter
    # returns: a sparse
    diagonals = []
    diagonals.append([2.] * n)
    diagonals.append([-1.] * (n - 1))
    diagonals.append([-1.] * (n - 1))
    A1 = sps.diags(diagonals, [0, 1, -1], shape=(n, n), format="csr")
    A1 = A1 * (n+1.)

    diagonals = []
    diagonals.append([.5] * (n-1))
    diagonals.append([-.5] * (n-1))
    A2 = sps.diags(diagonals, [1, -1], shape=(n, n), format="csr")
    A2 = A2 * gamma

    A = A1 + A2

    return A


def generateB(n: int) -> np.ndarray:
    # n: points in the grid
    # gamma: parameter
    # returns: a nx1 np array
    return np.ones((n, 1)) * (1 / (n + 1))