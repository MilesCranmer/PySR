import numpy as np
from pysr import pysr
X = np.random.randn(100, 5)

print("Test 1 - defaults; simple linear relation")
y = X[:, 0]
equations = pysr(X, y,
                 niterations=10)
print(equations)
assert equations.iloc[-1]['MSE'] < 1e-10

print("Test 2 - test custom operator")
y = X[:, 0]**2
equations = pysr(X, y,
                 unary_operators=["square(x) = x^2"], binary_operators=["plus"],
                 extra_sympy_mappings={'square': lambda x: x**2},
                 niterations=10)
print(equations)
assert equations.iloc[-1]['MSE'] < 1e-10

X = np.random.randn(100, 1)
y = X[:, 0] + 3.0
print("Test 3 - empty operator list, and single dimension input")
equations = pysr(X, y,
                 unary_operators=[], binary_operators=["plus"],
                 niterations=10)

print(equations)
assert equations.iloc[-1]['MSE'] < 1e-10
