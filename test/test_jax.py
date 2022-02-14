import unittest
import numpy as np
from pysr import sympy2jax, PySRRegressor
import pandas as pd
from jax import numpy as jnp
from jax import random
from jax import grad
import sympy


class TestJAX(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_sympy2jax(self):
        x, y, z = sympy.symbols("x y z")
        cosx = 1.0 * sympy.cos(x) + y
        key = random.PRNGKey(0)
        X = random.normal(key, (1000, 2))
        true = 1.0 * jnp.cos(X[:, 0]) + X[:, 1]
        f, params = sympy2jax(cosx, [x, y, z])
        self.assertTrue(jnp.all(jnp.isclose(f(X, params), true)).item())

    def test_pipeline(self):
        X = np.random.randn(100, 10)
        equations = pd.DataFrame(
            {
                "Equation": ["1.0", "cos(x1)", "square(cos(x1))"],
                "MSE": [1.0, 0.1, 1e-5],
                "Complexity": [1, 2, 3],
            }
        )

        equations["Complexity MSE Equation".split(" ")].to_csv(
            "equation_file.csv.bkup", sep="|"
        )

        model = PySRRegressor(
            equation_file="equation_file.csv",
            output_jax_format=True,
            variable_names="x1 x2 x3".split(" "),
        )

        model.selection = [1, 2, 3]
        model.n_features = 3
        model.using_pandas = False
        model.refresh()
        jformat = model.jax()

        np.testing.assert_almost_equal(
            np.array(jformat["callable"](jnp.array(X), jformat["parameters"])),
            np.square(np.cos(X[:, 1])),  # Select feature 1
            decimal=4,
        )
