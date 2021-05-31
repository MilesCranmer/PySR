import numpy as np
from pysr import sympy2torch
import torch
import sympy

print("Test Torch 1 - test export")
x, y, z = sympy.symbols('x y z')
cosx = 1.0 * sympy.cos(x) + y
X = torch.randn((1000, 2))
true = 1.0 * torch.cos(X[:, 0]) + X[:, 1]
torch_module = sympy2torch(cosx, [x, y, z])
assert np.all(np.isclose(torch_module(X).detach().numpy(), true.detach().numpy()))
