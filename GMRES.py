import numpy as np
import scipy as sp


class gmres:
    def __init__(self):
        self.M = None

    def mygmres(
        self, l: int, b: np.ndarray, x0: np.ndarray, n: int, M: np.ndarray, A: np.array
    ):
        # l: number of iterations
        # b: right hand side
        # x0: initial guess
        # n: number of unknowns
        # M: inner product matrix
        # A: matrix
        self.M = M
        assert l <= A.shape[0]

        r0 = b - A@x0
        Beta = np.linalg.norm(r0)
        v0 = r0 / Beta

    

    def dot(self, A: np.ndarray, B: np.ndarray):
        # A: matrix
        # B: matrix
        # returns: inner product of A and B 
        # with respect to the M inner product matrix
        return A.T @ self.M @ B
    
    def error_bound(self, A: np.ndarray, x: np.ndarray, b: np.ndarray):
        # A: matrix
        # x: solution
        # b: right hand side
        # returns: the error bound
        val = np.linalg.norm(b - A@x) / A.shape[0]
        self.error = val
        return val > 10**-6