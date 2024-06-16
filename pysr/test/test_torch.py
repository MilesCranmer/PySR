import unittest

import numpy as np
import pandas as pd
import sympy

from pysr import PySRRegressor, sympy2torch


class TestTorch(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

        # Need to import after juliacall:
        import torch

        self.torch = torch

    def test_sympy2torch(self):
        x, y, z = sympy.symbols("x y z")
        cosx = 1.0 * sympy.cos(x) + y

        X = self.torch.tensor(np.random.randn(1000, 3))
        true = 1.0 * self.torch.cos(X[:, 0]) + X[:, 1]
        torch_module = sympy2torch(cosx, [x, y, z])
        self.assertTrue(
            np.all(np.isclose(torch_module(X).detach().numpy(), true.detach().numpy()))
        )

    def test_pipeline_pandas(self):
        X = pd.DataFrame(np.random.randn(100, 10))
        y = np.ones(X.shape[0])
        model = PySRRegressor(
            progress=False,
            max_evals=10000,
            model_selection="accuracy",
            extra_sympy_mappings={},
            output_torch_format=True,
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
        tformat = model.pytorch()
        self.assertEqual(str(tformat), "_SingleSymPyModule(expression=cos(x1)**2)")

        np.testing.assert_almost_equal(
            tformat(self.torch.tensor(X.values)).detach().numpy(),
            np.square(np.cos(X.values[:, 1])),  # Selection 1st feature
            decimal=3,
        )

    def test_pipeline(self):
        X = np.random.randn(100, 10)
        y = np.ones(X.shape[0])
        model = PySRRegressor(
            progress=False,
            max_evals=10000,
            model_selection="accuracy",
            output_torch_format=True,
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

        tformat = model.pytorch()
        self.assertEqual(str(tformat), "_SingleSymPyModule(expression=cos(x1)**2)")

        np.testing.assert_almost_equal(
            tformat(self.torch.tensor(X)).detach().numpy(),
            np.square(np.cos(X[:, 1])),  # 2nd feature
            decimal=3,
        )

    def test_mod_mapping(self):
        x, y, z = sympy.symbols("x y z")
        expression = x**2 + sympy.atanh(sympy.Mod(y + 1, 2) - 1) * 3.2 * z

        module = sympy2torch(expression, [x, y, z])

        X = self.torch.rand(100, 3).float() * 10

        true_out = (
            X[:, 0] ** 2
            + self.torch.atanh(self.torch.fmod(X[:, 1] + 1, 2) - 1) * 3.2 * X[:, 2]
        )
        torch_out = module(X)

        np.testing.assert_array_almost_equal(
            true_out.detach(), torch_out.detach(), decimal=3
        )

    def test_custom_operator(self):
        X = np.random.randn(100, 3)
        y = np.ones(X.shape[0])
        model = PySRRegressor(
            progress=False,
            max_evals=10000,
            model_selection="accuracy",
            output_torch_format=True,
        )
        model.fit(X, y)

        equations = pd.DataFrame(
            {
                "Equation": ["1.0", "mycustomoperator(x1)"],
                "Loss": [1.0, 0.1],
                "Complexity": [1, 2],
            }
        )

        equations["Complexity Loss Equation".split(" ")].to_csv(
            "equation_file_custom_operator.csv.bkup"
        )

        model.set_params(
            equation_file="equation_file_custom_operator.csv",
            extra_sympy_mappings={"mycustomoperator": sympy.sin},
            extra_torch_mappings={"mycustomoperator": self.torch.sin},
        )
        model.refresh(checkpoint_file="equation_file_custom_operator.csv")
        self.assertEqual(str(model.sympy()), "sin(x1)")
        # Will automatically use the set global state from get_hof.

        tformat = model.pytorch()
        self.assertEqual(str(tformat), "_SingleSymPyModule(expression=sin(x1))")
        np.testing.assert_almost_equal(
            tformat(self.torch.tensor(X)).detach().numpy(),
            np.sin(X[:, 1]),
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
            extra_torch_mappings={"cos_approx": cos_approx},
            random_state=0,
            deterministic=True,
            procs=0,
            multithreading=False,
        )
        np.random.seed(0)
        model.fit(X.values, y.values)
        torch_module = model.pytorch()

        np_output = model.predict(X.values)

        torch_output = torch_module(self.torch.tensor(X.values)).detach().numpy()

        np.testing.assert_almost_equal(y.values, np_output, decimal=3)
        np.testing.assert_almost_equal(y.values, torch_output, decimal=3)


def runtests(just_tests=False):
    """Run all tests in test_torch.py."""
    tests = [TestTorch]
    if just_tests:
        return tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test in tests:
        suite.addTests(loader.loadTestsFromTestCase(test))
    runner = unittest.TextTestRunner()
    return runner.run(suite)
