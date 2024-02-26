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


class gmres:
    def __init__(self):
        self.M = None
        self.errors = []
        self.best_error = float('inf')
        self.r=None

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
        self.n = n
        assert l <= A.shape[0]
       

        # l corresponds to the number of iterations
        #start at 2, go up to l while error bound is not met
        m = 2
        xm = x0
        self.curr_xm = xm
        #lines one from the reading    
        
        while self.error_bound(A, xm, b) and m <= l:    
            #pregenerate empty matrices for later use
            r = b - A@xm
            Beta = np.linalg.norm(r)
            v0 = r / Beta
            W = np.zeros((n,m))
            V = np.zeros((n,m+1))
            V[:,0] = v0.reshape(-1)
            H = np.zeros((m+1,m))
            # the rest is pretty exactly the same as lines 2-10
            for j in range(m):
                W[:,j] = A@V[:,j]
                for i in range(j+1):
                    H[i,j] = self.dot(W[:,j],V[:,i])
                    W[:,j] -= H[i,j]*V[:,i]
                H[j+1,j] = self.norm(W[:,j])
                if H[j+1,j] == 0:
                    m = j
                    #I think this is correct, but I'm not sure
                    V = V[:,:j+1]
                    H = H[:j+1,:j]
                    break
                V[:,j+1] = W[:,j]/H[j+1,j]
            # unlike reading, H is defined but have to build e1
            e1 = np.zeros((m+1,1))
            e1[0,0] = 1
            #least squares solution
            ym = sp.linalg.lstsq(H,Beta*e1)[0]
            xm = xm + V[:,:-1]@ym
            self.curr_xm = xm

            #iterate up the m for higher l value
            m = m*2 if m*2 < l or m==l else l
        return xm, Beta
    
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
        if val < self.best_error:
            self.best_error = val
            self.best_x = self.curr_xm
        return val > 10**-6
    
    def norm(self, A):
        return np.sqrt((A.T @ self.M @ A).reshape(-1)[0])
    
    def phi(self, x, i):
        assert i < self.n
        assert x <= 1.00001 and x >= -0.00001
        h = 1/(self.n+1)
        l = (i)*h
        m = (i+1)*h
        r = (i+2)*h
        if x <= l or x >= r:
            return 0
        elif x < m:
            return (x-l)/h
        else:
            return (r-x)/h

    def sol_to_xy(self):
        u_coefs = self.best_x
        N = self.n * 25
        x = np.linspace(0, 1, N)
        y = np.zeros(N)
        for i in range(self.n):
            for j in range(N):
                y[j] += u_coefs[i] * self.phi(x[j], i)
        return x, y

# Specify parameters
n = 32  # Number of points in the grid
gamma = 1  # Parameter used in generateA
l = 12  # Number of iterations for the GMRES solver

# Generate matrix A
A = generateA(n, gamma)

# Generate right-hand side vector b
b = generateB(n)

# Initial guess for the solution
x0 = np.zeros((n, 1))

# Inner product matrix (identity matrix in this case)
M = np.eye(n)

# Create an instance of gmres
gmres_solver = gmres()

# Call the mygmres method to solve the linear system
xm, Beta = gmres_solver.mygmres(l, b, x0, n, M, A)

# Print the solution
print(xm)
print(Beta/n)

norm = gmres_solver.norm(A)
error= gmres_solver.error_bound(A,xm,b)

# Print the result
print("Norm:", norm)
print("Error Bound:", error)
