import numpy as np
from scipy.fftpack import dct

# def f(x):
#     return np.sin(1/x)

# x = np.linspace(-1,1,100)

def chebyshev(f,x):
    def chebyshev_nodes(n):
        """Compute Chebyshev nodes."""
        k = np.arange(1, n+1)
        return np.cos((2*k - 1) * np.pi / (2*n))

    def chebyshev_coefficients(f_values):
        """Compute Chebyshev coefficients using DCT."""
        n = len(f_values) - 1
        c = dct(f_values, type=1) / n
        c[0] /= 2
        return c

    def evaluate_chebyshev_polynomial(x, coefficients):
        """Evaluate Chebyshev polynomial at point x."""
        n = len(coefficients) - 1
        T = np.zeros((len(x), n+1))
        T[:, 0] = 1
        T[:, 1] = x
        for k in range(2, n+1):
            T[:, k] = 2 * x * T[:, k-1] - T[:, k-2]
        return np.dot(T, coefficients)

    # Define the function to approximate
    #f = lambda x: np.sin(1/x)

    # Define the interval
    a, b = x[0], x[-1]

    # Choose the degree of polynomial
    degree = 1000

    # Compute Chebyshev nodes
    nodes = chebyshev_nodes(degree)

    # Evaluate the function at Chebyshev nodes
    f_values = f((b - a) / 2 * nodes + (a + b) / 2)

    # Compute Chebyshev coefficients
    coefficients = chebyshev_coefficients(f_values)

    # Evaluate approximation at these points
    approximation = evaluate_chebyshev_polynomial((2 * x - (a + b)) / (b - a), coefficients)
    derivative = np.gradient(approximation,x)

    # Plot the original function and its approximation
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(8, 6))
    # plt.plot(x_vals, f(x_vals), label='f(x)', color='blue')
    # plt.plot(x_vals, approximation, label='Chebyshev Approximation', linestyle='--', color='red')
    # plt.title('Chebyshev Approximation of a function')
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(8, 6))
    # plt.plot(x, np.gradient(f(x),x), label='Derivative', color='blue')
    # plt.plot(x, derivative, label='Derivative Approximation', linestyle='--', color='red')
    # plt.title('Chebyshev Approximation of derivative')
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return derivative

#result = chebyshev(f,x)