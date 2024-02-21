import numpy as np
import scipy.sparse as sp


def generateA(n: int, gamma: float) -> sp.csr_matrix:
    # n: points in the grid
    # gamma: parameter
    # returns: a sparse matrix
    pass


def generateB(n: int) -> np.ndarray:
    # n: points in the grid
    # gamma: parameter
    # returns: a nx1 np array
    return np.ones((n, 1)) * (1 / (n + 2))
