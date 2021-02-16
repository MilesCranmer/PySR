import numpy as np
from pysr import pysr, best

# Dataset
X = 2*np.random.randn(100, 5)
y = 2*np.cos(X[:, 3]) + X[:, 0]**2 - 2

# Learn equations
equations = pysr(X, y, niterations=5,
    binary_operators=["plus", "mult"],
    unary_operators=[
      "cos", "exp", "sin", #Pre-defined library of operators (see https://pysr.readthedocs.io/en/latest/docs/operators/)
      "inv(x) = 1/x"],
    loss='loss(x, y) = abs(x - y)', # Custom loss function
    julia_project="../SymbolicRegression.jl") # Define your own operator! (Julia syntax)

...# (you can use ctl-c to exit early)

print(best(equations))
