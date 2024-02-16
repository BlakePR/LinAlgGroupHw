import numpy as np
import scipy as sp


class gmres:
    def __init__(self):
        pass

    def mygmres(
        self, l: int, b: np.ndarray, x0: np.ndarray, n: int, M: np.ndarray, A: np.array
    ):
        # l: number of iterations
        # b: right hand side
        # x0: initial guess
        # n: number of unknowns
        # M: preconditioner
        # A: matrix

        pass
        # TODO: Implement GMRES
        # make a change


# example numpy stuff, delete later
b = np.array([1, 0])
A = np.array([[1, 3], [2, 6]])
matmul = A @ A
mat_vec = A @ b
print(mat_vec)
# np often makes vectors (n,) instead of (n,1)
# so reshape them like so
b = b.reshape(2, 1)
