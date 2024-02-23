import numpy as np
import scipy as sp


class gmres:
    def __init__(self):
        self.M = None
        self.errors = []

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
        m = 2
        xm = x0
        while self.error_bound(A, xm, b) and m <= l:
            W = np.zeros((n,m))
            V = np.zeros((n,m+1))
            V[:,0] = v0
            H = np.zeros((m+1,m))
            for j in range(m):
                W[:,j] = A@V[:j]
                for i in range(j):
                    H[i,j] = self.dot(W[:,j],V[:,i])
                    W[:,j] = H[i,j]@v[:,j]
                H[j+1,j] = self.norm(W[:,j])
                if H[j+1,j] == 0:
                    m = j
                    break
                V[:,j+1] = W[:,j]/H[j+1,j]
            e1 = np.zeros(m+1,1)
            e1[0,0] = 1
            ym = np.linalg.solve(H,Beta*e1)
            xm = x0 + V@ym

            m *=2
        return xm
    
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
        val = self.norm(b - A@x) / A.shape[0]
        self.errors.append(val)
        return val > 10**-6
    
    def norm(self, A):
        return A.T @ self.M @ A