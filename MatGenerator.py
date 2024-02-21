import numpy as np
import scipy.sparse as sps


def generateA(n: int, gamma: float) -> sps.csr_matrix:
    # n: points in the grid
    # gamma: parameter
    # returns: a sparse
    diagonals = [[]]

    A1 = sps.diags()
    A2 = None

    A = A1 + A2
    return A


def generateB(n: int) -> np.ndarray:
    # n: points in the grid
    # gamma: parameter
    # returns: a nx1 np array
    return np.ones((n, 1)) * (1 / (n + 2))
