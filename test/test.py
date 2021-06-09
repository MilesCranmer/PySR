import unittest
from unittest.mock import patch
import numpy as np
from pysr import pysr, get_hof, best, best_tex, best_callable, best_row
from pysr.sr import run_feature_selection, _handle_feature_selection, _yesno
import sympy
from sympy import lambdify
import pandas as pd


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.default_test_kwargs = dict(
            niterations=10,
            populations=4,
            user_input=False,
            annealing=True,
            useFrequency=False,
        )
        np.random.seed(0)
        self.X = np.random.randn(100, 5)

    def test_linear_relation(self):
        y = self.X[:, 0]
        equations = pysr(self.X, y, **self.default_test_kwargs)
        print(equations)
        self.assertLessEqual(equations.iloc[-1]["MSE"], 1e-4)

    def test_multioutput_custom_operator(self):
        y = self.X[:, [0, 1]] ** 2
        equations = pysr(
            self.X,
            y,
            unary_operators=["sq(x) = x^2"],
            binary_operators=["plus"],
            extra_sympy_mappings={"sq": lambda x: x ** 2},
            **self.default_test_kwargs,
            procs=0,
        )
        print(equations)
        self.assertLessEqual(equations[0].iloc[-1]["MSE"], 1e-4)
        self.assertLessEqual(equations[1].iloc[-1]["MSE"], 1e-4)

    def test_multioutput_weighted_with_callable(self):
        y = self.X[:, [0, 1]] ** 2
        w = np.random.rand(*y.shape)
        w[w < 0.5] = 0.0
        w[w >= 0.5] = 1.0

        # Double equation when weights are 0:
        y = (2 - w) * y
        # Thus, pysr needs to use the weights to find the right equation!

        pysr(
            self.X,
            y,
            weights=w,
            unary_operators=["sq(x) = x^2"],
            binary_operators=["plus"],
            extra_sympy_mappings={"sq": lambda x: x ** 2},
            **self.default_test_kwargs,
            procs=0,
        )

        np.testing.assert_almost_equal(
            best_callable()[0](self.X), self.X[:, 0] ** 2, decimal=4
        )
        np.testing.assert_almost_equal(
            best_callable()[1](self.X), self.X[:, 1] ** 2, decimal=4
        )

    def test_empty_operators_single_input(self):
        X = np.random.randn(100, 1)
        y = X[:, 0] + 3.0
        equations = pysr(
            X,
            y,
            unary_operators=[],
            binary_operators=["plus"],
            **self.default_test_kwargs,
        )

        self.assertLessEqual(equations.iloc[-1]["MSE"], 1e-4)

    def test_noisy(self):

        np.random.seed(1)
        y = self.X[:, [0, 1]] ** 2 + np.random.randn(self.X.shape[0], 1) * 0.05
        equations = pysr(
            self.X,
            y,
            unary_operators=["sq(x) = x^2"],
            binary_operators=["plus"],
            extra_sympy_mappings={"sq": lambda x: x ** 2},
            **self.default_test_kwargs,
            procs=0,
            denoise=True,
        )
        self.assertLessEqual(best_row(equations=equations)[0]["MSE"], 1e-2)
        self.assertLessEqual(best_row(equations=equations)[1]["MSE"], 1e-2)


class TestBest(unittest.TestCase):
    def setUp(self):
        equations = pd.DataFrame(
            {
                "Equation": ["1.0", "cos(x0)", "square(cos(x0))"],
                "MSE": [1.0, 0.1, 1e-5],
                "Complexity": [1, 2, 3],
            }
        )

        equations["Complexity MSE Equation".split(" ")].to_csv(
            "equation_file.csv.bkup", sep="|"
        )

        self.equations = get_hof(
            "equation_file.csv",
            n_features=2,
            variables_names="x0 x1".split(" "),
            extra_sympy_mappings={},
            output_jax_format=False,
            multioutput=False,
            nout=1,
        )

    def test_best(self):
        self.assertEqual(best(self.equations), sympy.cos(sympy.Symbol("x0")) ** 2)
        self.assertEqual(best(), sympy.cos(sympy.Symbol("x0")) ** 2)

    def test_best_tex(self):
        self.assertEqual(best_tex(self.equations), "\\cos^{2}{\\left(x_{0} \\right)}")
        self.assertEqual(best_tex(), "\\cos^{2}{\\left(x_{0} \\right)}")

    def test_best_lambda(self):
        X = np.random.randn(10, 2)
        y = np.cos(X[:, 0]) ** 2
        for f in [best_callable(), best_callable(self.equations)]:
            np.testing.assert_almost_equal(f(X), y, decimal=4)


class TestFeatureSelection(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_feature_selection(self):
        X = np.random.randn(20000, 5)
        y = X[:, 2] ** 2 + X[:, 3] ** 2
        selected = run_feature_selection(X, y, select_k_features=2)
        self.assertEqual(sorted(selected), [2, 3])

    def test_feature_selection_handler(self):
        X = np.random.randn(20000, 5)
        y = X[:, 2] ** 2 + X[:, 3] ** 2
        var_names = [f"x{i}" for i in range(5)]
        selected_X, selected_var_names, selection = _handle_feature_selection(
            X,
            select_k_features=2,
            use_custom_variable_names=True,
            variable_names=var_names,
            y=y,
        )
        self.assertTrue((2 in selection) and (3 in selection))
        self.assertEqual(set(selected_var_names), set("x2 x3".split(" ")))
        np.testing.assert_array_equal(
            np.sort(selected_X, axis=1), np.sort(X[:, [2, 3]], axis=1)
        )


class TestHelperFunctions(unittest.TestCase):
    @patch("builtins.input", side_effect=["y", "n"])
    def test_yesno(self, mock_input):
        # Assert that the yes/no function correctly deals with y/n
        self.assertEqual(_yesno("Test"), True)
        self.assertEqual(_yesno("Test"), False)
