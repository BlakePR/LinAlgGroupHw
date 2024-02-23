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

        # l corresponds to the number of iterations
        #start at 2, go up to l while error bound is not met
        m = 2
        xm = x0
        while self.error_bound(A, xm, b) and m <= l:
            #lines one from the reading
            r = b - A@x0
            Beta = np.linalg.norm(r)
            v0 = r / Beta
            #pregenerate empyt matrices for later use
            W = np.zeros((n,m))
            V = np.zeros((n,m))
            V[:,0] = v0.reshape(-1)
            H = np.zeros((m+1,m))
            # the rest is pretty exactly the same as lines 2-10
            for j in range(m):
                W[:,j] = A@V[:,j]
                for i in range(j):
                    H[i,j] = self.dot(W[:,j],V[:,i])
                    W[:,j] = H[i,j]*V[:,j]
                H[j+1,j] = self.norm(W[:,j])
                if H[j+1,j] == 0:
                    m = j
                    break
                V[:,j+1] = W[:,j]/H[j+1,j]
            # unlike reading, H is defined but have to build e1
            e1 = np.zeros((m+1,1))
            e1[0,0] = 1
            #least squares solution
            ym = np.linalg.lstsq(H,Beta*e1)[0]
            xm = xm + V@ym

            #iterate up the m for higher l value
            m = m*2 if m*2 < l else l
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