#requires pip install num-dual
import numpy as np

def dual(f, x):
    from num_dual import Dual64
    temp = np.array([Dual64(xi, 1) for xi in x])
    fx = f(temp)
    derivs = np.array([fxi.first_derivative for fxi in fx])
    return derivs

# x = np.array([.5,1.,1.5,2.])
# x = x * np.pi

# print(x)

# dx = dual(np.sin, x)
# print(dx)
# # Expected output:
# print(np.cos(x))