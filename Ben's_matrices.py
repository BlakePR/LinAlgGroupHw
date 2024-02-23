""" These are the functions that generate the symbolic versions of matrices A1 
and A2 as well as the vector b. """

import sympy as sp

def generate_A1(n):
    # Define symbols
    x = sp.Symbol('x')
    
    # Define symbolic functions φ_i(x)
    phis = [sp.Function('phi_' + str(i))(x) for i in range(1, n+1)]
    
    # Compute derivatives φ_i'(x)
    phi_primes = [phi.diff(x) for phi in phis]
    
    # Initialize the matrix A1
    A1 = sp.zeros(n)
    
    # Compute integrals and fill the matrix
    for i in range(n):
        for j in range(n):
            integral = sp.integrate(phi_primes[i] * phi_primes[j], x)
            A1[i, j] = integral
    
    # Display the matrix
    print("Matrix A1:")
    sp.pprint(A1)
    
    return A1
    
def generate_A2(n,ɣ):
    # Define symbols
    x = sp.Symbol('x')

    # Define symbolic functions φ_i(x)
    phis = [sp.Function('phi_' + str(i))(x) for i in range(1, n+1)]

    # Compute derivatives φ_i'(x)
    phi_primes = [phi.diff(x) for phi in phis]

    # Initialize the matrix A2
    A2 = sp.zeros(n)

    # Compute integrals and fill the matrix
    for i in range(n):
        for j in range(n):
            integral = sp.integrate(phis[i] * phi_primes[j], x)
#Use the following line to output the diagonal elements without evaluating the integral
            #A2[i, j] = sp.Integral(phis[i] * phi_primes[j], x) 
            A2[i, j] = integral
    A2 = ɣ*A2
    
    # Display the matrix
    print("Matrix A2:")
    sp.pprint(A2)
    
    return A2
    
def generate_b(n):
    x = sp.Symbol('x')
        
    # Define symbolic functions φ_j(x)
    phis = [sp.Function('phi_' + str(i))(x) for i in range(1, n+1)]

    # Initialize the vector
    vector = sp.zeros(n, 1)

    # Compute integrals and fill the vector
    for i in range(n):
        integral = sp.integrate(phis[i], x)
        vector[i, 0] = sp.Integral(phis[i], x)
        
    print("Vector b:")
    sp.pprint(vector)  
    
    return vector

