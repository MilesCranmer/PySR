import numpy as np
from pysr import pysr, best, get_hof

# Dataset
X = 2*np.random.randn(100, 5)
y = 2*np.cos(X[:, 3]) + X[:, 0]**2 - 2

# Learn equations
equations = pysr(X, y, niterations=5,
                binary_operators=["plus", "mult"],
                unary_operators=["cos", "exp", "sin"])

...  # (you can use ctl-c to exit early)

print(best(equations))