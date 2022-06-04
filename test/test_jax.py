import unittest
import numpy as np
from pysr import sympy2jax, PySRRegressor
import pandas as pd
from jax import numpy as jnp
from jax import random
import sympy
from functools import partial


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

    def test_pipeline_pandas(self):
        X = pd.DataFrame(np.random.randn(100, 10))
        y = np.ones(X.shape[0])
        model = PySRRegressor(
            progress=False,
            max_evals=10000,
            output_jax_format=True,
        )
        model.fit(X, y)

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

        model.refresh(checkpoint_file="equation_file.csv")
        jformat = model.jax()

        np.testing.assert_almost_equal(
            np.array(jformat["callable"](jnp.array(X), jformat["parameters"])),
            np.square(np.cos(X.values[:, 1])),  # Select feature 1
            decimal=4,
        )

    def test_pipeline(self):
        X = np.random.randn(100, 10)
        y = np.ones(X.shape[0])
        model = PySRRegressor(progress=False, max_evals=10000, output_jax_format=True)
        model.fit(X, y)

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

        model.refresh(checkpoint_file="equation_file.csv")
        jformat = model.jax()

        np.testing.assert_almost_equal(
            np.array(jformat["callable"](jnp.array(X), jformat["parameters"])),
            np.square(np.cos(X[:, 1])),  # Select feature 1
            decimal=4,
        )

    def test_feature_selection(self):
        X = pd.DataFrame({f"k{i}": np.random.randn(1000) for i in range(10, 21)})
        y = X["k15"] ** 2 + np.cos(X["k20"])

        model = PySRRegressor(
            progress=False,
            unary_operators=["cos"],
            select_k_features=3,
            early_stop_condition=1e-5,
        )
        model.fit(X.values, y.values)
        f, parameters = model.jax().values()

        np_prediction = model.predict
        jax_prediction = partial(f, parameters=parameters)

        np_output = np_prediction(X.values)
        jax_output = jax_prediction(X.values)

        np.testing.assert_almost_equal(np_output, jax_output, decimal=4)
