import unittest
import numpy as np
import pandas as pd
import sympy
from functools import partial

from .. import sympy2jax, PySRRegressor


class TestJAX(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_sympy2jax(self):
        from jax import numpy as jnp
        from jax import random

        x, y, z = sympy.symbols("x y z")
        cosx = 1.0 * sympy.cos(x) + y
        key = random.PRNGKey(0)
        X = random.normal(key, (1000, 2))
        true = 1.0 * jnp.cos(X[:, 0]) + X[:, 1]
        f, params = sympy2jax(cosx, [x, y, z])
        self.assertTrue(jnp.all(jnp.isclose(f(X, params), true)).item())

    def test_pipeline_pandas(self):
        from jax import numpy as jnp

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
                "Loss": [1.0, 0.1, 1e-5],
                "Complexity": [1, 2, 3],
            }
        )

        equations["Complexity Loss Equation".split(" ")].to_csv(
            "equation_file.csv.bkup"
        )

        model.refresh(checkpoint_file="equation_file.csv")
        jformat = model.jax()

        np.testing.assert_almost_equal(
            np.array(jformat["callable"](jnp.array(X), jformat["parameters"])),
            np.square(np.cos(X.values[:, 1])),  # Select feature 1
            decimal=3,
        )

    def test_pipeline(self):
        from jax import numpy as jnp

        X = np.random.randn(100, 10)
        y = np.ones(X.shape[0])
        model = PySRRegressor(progress=False, max_evals=10000, output_jax_format=True)
        model.fit(X, y)

        equations = pd.DataFrame(
            {
                "Equation": ["1.0", "cos(x1)", "square(cos(x1))"],
                "Loss": [1.0, 0.1, 1e-5],
                "Complexity": [1, 2, 3],
            }
        )

        equations["Complexity Loss Equation".split(" ")].to_csv(
            "equation_file.csv.bkup"
        )

        model.refresh(checkpoint_file="equation_file.csv")
        jformat = model.jax()

        np.testing.assert_almost_equal(
            np.array(jformat["callable"](jnp.array(X), jformat["parameters"])),
            np.square(np.cos(X[:, 1])),  # Select feature 1
            decimal=3,
        )

    def test_feature_selection_custom_operators(self):
        rstate = np.random.RandomState(0)
        X = pd.DataFrame({f"k{i}": rstate.randn(2000) for i in range(10, 21)})
        cos_approx = lambda x: 1 - (x**2) / 2 + (x**4) / 24 + (x**6) / 720
        y = X["k15"] ** 2 + 2 * cos_approx(X["k20"])

        model = PySRRegressor(
            progress=False,
            unary_operators=["cos_approx(x) = 1 - x^2 / 2 + x^4 / 24 + x^6 / 720"],
            select_k_features=3,
            maxsize=10,
            early_stop_condition=1e-5,
            extra_sympy_mappings={"cos_approx": cos_approx},
            extra_jax_mappings={
                "cos_approx": "(lambda x: 1 - x**2 / 2 + x**4 / 24 + x**6 / 720)"
            },
            random_state=0,
            deterministic=True,
            procs=0,
            multithreading=False,
        )
        np.random.seed(0)
        model.fit(X.values, y.values)
        f, parameters = model.jax().values()

        np_prediction = model.predict
        jax_prediction = partial(f, parameters=parameters)

        np_output = np_prediction(X.values)
        jax_output = jax_prediction(X.values)

        np.testing.assert_almost_equal(y.values, np_output, decimal=3)
        np.testing.assert_almost_equal(y.values, jax_output, decimal=3)


def runtests():
    """Run all tests in test_jax.py."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestJAX))
    runner = unittest.TextTestRunner()
    return runner.run(suite)
