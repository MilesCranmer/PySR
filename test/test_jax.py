import numpy as np
from pysr import pysr, sympy2jax
from jax import numpy as jnp
from jax import random
from jax import grad
import sympy

print("Test JAX 1 - test export")
x, y, z = sympy.symbols('x y z')
cosx = 1.0 * sympy.cos(x) + y
key = random.PRNGKey(0)
X = random.normal(key, (1000, 2))
true = 1.0 * jnp.cos(X[:, 0]) + X[:, 1]
f, params = sympy2jax(cosx, [x, y, z])
assert jnp.all(jnp.isclose(f(X, params), true)).item()
