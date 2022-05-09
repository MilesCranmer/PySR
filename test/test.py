import inspect
import unittest
from unittest.mock import patch
import numpy as np
from pysr import PySRRegressor
from pysr.sr import run_feature_selection, _handle_feature_selection
import sympy
from sympy import lambdify
import pandas as pd


class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Using inspect,
        # get default niterations from PySRRegressor, and double them:
        default_niterations = (
            inspect.signature(PySRRegressor.__init__).parameters["niterations"].default
        )
        default_populations = (
            inspect.signature(PySRRegressor.__init__).parameters["populations"].default
        )
        self.default_test_kwargs = dict(
            model_selection="accuracy",
            niterations=default_niterations * 2,
            populations=default_populations * 2,
        )
        self.rstate = np.random.RandomState(0)
        self.X = self.rstate.randn(100, 5)

    def test_linear_relation(self):
        y = self.X[:, 0]
        model = PySRRegressor(**self.default_test_kwargs)
        model.fit(self.X, y)
        print(model.equations)
        self.assertLessEqual(model.get_best()["loss"], 1e-4)

    def test_multiprocessing(self):
        y = self.X[:, 0]
        model = PySRRegressor(**self.default_test_kwargs, procs=2, multithreading=False)
        model.fit(self.X, y)
        print(model.equations)
        self.assertLessEqual(model.equations.iloc[-1]["loss"], 1e-4)

    def test_multioutput_custom_operator_quiet(self):
        y = self.X[:, [0, 1]] ** 2
        model = PySRRegressor(
            unary_operators=["square_op(x) = x^2"],
            extra_sympy_mappings={"square_op": lambda x: x**2},
            binary_operators=["plus"],
            verbosity=0,
            **self.default_test_kwargs,
            procs=0,
            # Test custom operators with constraints:
            nested_constraints={"square_op": {"square_op": 3}},
            constraints={"square_op": 10},
        )
        model.fit(self.X, y)
        equations = model.equations
        print(equations)
        self.assertIn("square_op", model.equations[0].iloc[-1]["equation"])
        self.assertLessEqual(equations[0].iloc[-1]["loss"], 1e-4)
        self.assertLessEqual(equations[1].iloc[-1]["loss"], 1e-4)

        test_y1 = model.predict(self.X)
        test_y2 = model.predict(self.X, index=[-1, -1])

        mse1 = np.average((test_y1 - y) ** 2)
        mse2 = np.average((test_y2 - y) ** 2)

        self.assertLessEqual(mse1, 1e-4)
        self.assertLessEqual(mse2, 1e-4)

        bad_y = model.predict(self.X, index=[0, 0])
        bad_mse = np.average((bad_y - y) ** 2)
        self.assertGreater(bad_mse, 1e-4)

    def test_multioutput_weighted_with_callable_temp_equation(self):
        X = self.X.copy()
        y = X[:, [0, 1]] ** 2
        w = self.rstate.rand(*y.shape)
        w[w < 0.5] = 0.0
        w[w >= 0.5] = 1.0

        # Double equation when weights are 0:
        y = (2 - w) * y
        # Thus, pysr needs to use the weights to find the right equation!

        model = PySRRegressor(
            unary_operators=["sq(x) = x^2"],
            binary_operators=["plus"],
            extra_sympy_mappings={"sq": lambda x: x**2},
            **self.default_test_kwargs,
            procs=0,
            temp_equation_file=True,
            delete_tempfiles=False,
        )
        model.fit(X.copy(), y, weights=w)

        np.testing.assert_almost_equal(
            model.predict(X.copy())[:, 0], X[:, 0] ** 2, decimal=4
        )
        np.testing.assert_almost_equal(
            model.predict(X.copy())[:, 1], X[:, 1] ** 2, decimal=4
        )

    def test_empty_operators_single_input_multirun(self):
        X = self.rstate.randn(100, 1)
        y = X[:, 0] + 3.0
        regressor = PySRRegressor(
            unary_operators=[],
            binary_operators=["plus"],
            **self.default_test_kwargs,
        )
        self.assertTrue("None" in regressor.__repr__())
        regressor.fit(X, y)
        self.assertTrue("None" not in regressor.__repr__())
        self.assertTrue(">>>>" in regressor.__repr__())

        self.assertLessEqual(regressor.equations.iloc[-1]["loss"], 1e-4)
        np.testing.assert_almost_equal(regressor.predict(X), y, decimal=1)

        # Test if repeated fit works:
        regressor.set_params(niterations=0)
        regressor.fit(X, y)

        self.assertLessEqual(regressor.equations.iloc[-1]["loss"], 1e-4)
        np.testing.assert_almost_equal(regressor.predict(X), y, decimal=1)

        # Tweak model selection:
        regressor.set_params(model_selection="best")
        self.assertEqual(regressor.get_params()["model_selection"], "best")
        self.assertTrue("None" not in regressor.__repr__())
        self.assertTrue(">>>>" in regressor.__repr__())

    def test_noisy(self):

        y = self.X[:, [0, 1]] ** 2 + self.rstate.randn(self.X.shape[0], 1) * 0.05
        model = PySRRegressor(
            # Test that passing a single operator works:
            unary_operators="sq(x) = x^2",
            binary_operators="plus",
            extra_sympy_mappings={"sq": lambda x: x**2},
            **self.default_test_kwargs,
            procs=0,
            denoise=True,
        )
        model.fit(self.X, y)
        self.assertLessEqual(model.get_best()[1]["loss"], 1e-2)
        self.assertLessEqual(model.get_best()[1]["loss"], 1e-2)

    def test_pandas_resample_with_nested_constraints(self):
        X = pd.DataFrame(
            {
                "T": self.rstate.randn(500),
                "x": self.rstate.randn(500),
                "unused_feature": self.rstate.randn(500),
            }
        )
        true_fn = lambda x: np.array(x["T"] + x["x"] ** 2 + 1.323837)
        y = true_fn(X)
        noise = self.rstate.randn(500) * 0.01
        y = y + noise
        # We also test y as a pandas array:
        y = pd.Series(y)
        # Resampled array is a different order of features:
        Xresampled = pd.DataFrame(
            {
                "unused_feature": self.rstate.randn(100),
                "x": self.rstate.randn(100),
                "T": self.rstate.randn(100),
            }
        )
        model = PySRRegressor(
            unary_operators=[],
            binary_operators=["+", "*", "/", "-"],
            **self.default_test_kwargs,
            Xresampled=Xresampled,
            denoise=True,
            select_k_features=2,
            nested_constraints={"/": {"+": 1, "-": 1}, "+": {"*": 4}},
        )
        model.fit(X, y)
        self.assertNotIn("unused_feature", model.latex())
        self.assertIn("T", model.latex())
        self.assertIn("x", model.latex())
        self.assertLessEqual(model.get_best()["loss"], 1e-1)
        fn = model.get_best()["lambda_format"]
        self.assertListEqual(list(sorted(fn._selection)), [0, 1])
        X2 = pd.DataFrame(
            {
                "T": self.rstate.randn(100),
                "unused_feature": self.rstate.randn(100),
                "x": self.rstate.randn(100),
            }
        )
        self.assertLess(np.average((fn(X2) - true_fn(X2)) ** 2), 1e-1)
        self.assertLess(np.average((model.predict(X2) - true_fn(X2)) ** 2), 1e-1)


class TestBest(unittest.TestCase):
    def setUp(self):
        equations = pd.DataFrame(
            {
                "equation": ["1.0", "cos(x0)", "square(cos(x0))"],
                "loss": [1.0, 0.1, 1e-5],
                "complexity": [1, 2, 3],
            }
        )

        equations["complexity loss equation".split(" ")].to_csv(
            "equation_file.csv.bkup", sep="|"
        )

        self.model = PySRRegressor(
            equation_file="equation_file.csv",
            variable_names="x0 x1".split(" "),
            extra_sympy_mappings={},
            output_jax_format=False,
            model_selection="accuracy",
        )
        self.model.n_features = 2
        self.model.refresh()
        self.equations = self.model.equations
        self.rstate = np.random.RandomState(0)

    def test_best(self):
        self.assertEqual(self.model.sympy(), sympy.cos(sympy.Symbol("x0")) ** 2)

    def test_index_selection(self):
        self.assertEqual(self.model.sympy(-1), sympy.cos(sympy.Symbol("x0")) ** 2)
        self.assertEqual(self.model.sympy(2), sympy.cos(sympy.Symbol("x0")) ** 2)
        self.assertEqual(self.model.sympy(1), sympy.cos(sympy.Symbol("x0")))
        self.assertEqual(self.model.sympy(0), 1.0)

    def test_best_tex(self):
        self.assertEqual(self.model.latex(), "\\cos^{2}{\\left(x_{0} \\right)}")

    def test_best_lambda(self):
        X = self.rstate.randn(10, 2)
        y = np.cos(X[:, 0]) ** 2
        for f in [self.model.predict, self.equations.iloc[-1]["lambda_format"]]:
            np.testing.assert_almost_equal(f(X), y, decimal=4)


class TestFeatureSelection(unittest.TestCase):
    def setUp(self):
        self.rstate = np.random.RandomState(0)

    def test_feature_selection(self):
        X = self.rstate.randn(20000, 5)
        y = X[:, 2] ** 2 + X[:, 3] ** 2
        selected = run_feature_selection(X, y, select_k_features=2)
        self.assertEqual(sorted(selected), [2, 3])

    def test_feature_selection_handler(self):
        X = self.rstate.randn(20000, 5)
        y = X[:, 2] ** 2 + X[:, 3] ** 2
        var_names = [f"x{i}" for i in range(5)]
        selected_X, selection = _handle_feature_selection(
            X,
            select_k_features=2,
            variable_names=var_names,
            y=y,
        )
        self.assertTrue((2 in selection) and (3 in selection))
        selected_var_names = [var_names[i] for i in selection]
        self.assertEqual(set(selected_var_names), set("x2 x3".split(" ")))
        np.testing.assert_array_equal(
            np.sort(selected_X, axis=1), np.sort(X[:, [2, 3]], axis=1)
        )


class TestMiscellaneous(unittest.TestCase):
    """Test miscellaneous functions."""

    def test_deprecation(self):
        # Ensure that deprecation works as expected, with a warning,
        # and sets the correct value.
        with self.assertWarns(UserWarning):
            model = PySRRegressor(fractionReplaced=0.2)
        # This is a deprecated parameter, so we should get a warning.

        # The correct value should be set:
        self.assertEqual(model.params["fraction_replaced"], 0.2)
