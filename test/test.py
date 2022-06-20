import os
import traceback
import inspect
import unittest
import numpy as np
from sklearn import model_selection
from pysr import PySRRegressor
from pysr.sr import run_feature_selection, _handle_feature_selection
from sklearn.utils.estimator_checks import check_estimator
import sympy
import pandas as pd
import warnings
import pickle as pkl
import tempfile

DEFAULT_PARAMS = inspect.signature(PySRRegressor.__init__).parameters
DEFAULT_NITERATIONS = DEFAULT_PARAMS["niterations"].default
DEFAULT_POPULATIONS = DEFAULT_PARAMS["populations"].default
DEFAULT_NCYCLES = DEFAULT_PARAMS["ncyclesperiteration"].default


class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Using inspect,
        # get default niterations from PySRRegressor, and double them:
        self.default_test_kwargs = dict(
            progress=False,
            model_selection="accuracy",
            niterations=DEFAULT_NITERATIONS * 2,
            populations=DEFAULT_POPULATIONS * 2,
            temp_equation_file=True,
        )
        self.rstate = np.random.RandomState(0)
        self.X = self.rstate.randn(100, 5)

    def test_linear_relation(self):
        y = self.X[:, 0]
        model = PySRRegressor(
            **self.default_test_kwargs,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity == 1",
        )
        model.fit(self.X, y)
        print(model.equations_)
        self.assertLessEqual(model.get_best()["loss"], 1e-4)

    def test_linear_relation_weighted(self):
        y = self.X[:, 0]
        weights = np.ones_like(y)
        model = PySRRegressor(
            **self.default_test_kwargs,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity == 1",
        )
        model.fit(self.X, y, weights=weights)
        print(model.equations_)
        self.assertLessEqual(model.get_best()["loss"], 1e-4)

    def test_multiprocessing(self):
        y = self.X[:, 0]
        model = PySRRegressor(
            **self.default_test_kwargs,
            procs=2,
            multithreading=False,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity == 1",
        )
        model.fit(self.X, y)
        print(model.equations_)
        self.assertLessEqual(model.equations_.iloc[-1]["loss"], 1e-4)

    def test_multioutput_custom_operator_quiet_custom_complexity(self):
        y = self.X[:, [0, 1]] ** 2
        model = PySRRegressor(
            unary_operators=["square_op(x) = x^2"],
            extra_sympy_mappings={"square_op": lambda x: x**2},
            complexity_of_operators={"square_op": 2, "plus": 1},
            binary_operators=["plus"],
            verbosity=0,
            **self.default_test_kwargs,
            procs=0,
            # Test custom operators with constraints:
            nested_constraints={"square_op": {"square_op": 3}},
            constraints={"square_op": 10},
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity == 3",
        )
        model.fit(self.X, y)
        equations = model.equations_
        print(equations)
        self.assertIn("square_op", model.equations_[0].iloc[-1]["equation"])
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
            delete_tempfiles=False,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity == 2",
        )
        model.fit(X.copy(), y, weights=w)

        # These tests are flaky, so don't fail test:
        try:
            np.testing.assert_almost_equal(
                model.predict(X.copy())[:, 0], X[:, 0] ** 2, decimal=4
            )
        except AssertionError:
            print("Error in test_multioutput_weighted_with_callable_temp_equation")
            print("Model equations: ", model.sympy()[0])
            print("True equation: x0^2")

        try:
            np.testing.assert_almost_equal(
                model.predict(X.copy())[:, 1], X[:, 1] ** 2, decimal=4
            )
        except AssertionError:
            print("Error in test_multioutput_weighted_with_callable_temp_equation")
            print("Model equations: ", model.sympy()[1])
            print("True equation: x1^2")

    def test_empty_operators_single_input_warm_start(self):
        X = self.rstate.randn(100, 1)
        y = X[:, 0] + 3.0
        regressor = PySRRegressor(
            unary_operators=[],
            binary_operators=["plus"],
            **self.default_test_kwargs,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity == 3",
        )
        self.assertTrue("None" in regressor.__repr__())
        regressor.fit(X, y)
        self.assertTrue("None" not in regressor.__repr__())
        self.assertTrue(">>>>" in regressor.__repr__())

        self.assertLessEqual(regressor.equations_.iloc[-1]["loss"], 1e-4)
        np.testing.assert_almost_equal(regressor.predict(X), y, decimal=1)

        # Test if repeated fit works:
        regressor.set_params(
            niterations=1,
            ncyclesperiteration=2,
            warm_start=True,
            early_stop_condition=None,
        )
        # This should exit almost immediately, and use the old equations
        regressor.fit(X, y)

        self.assertLessEqual(regressor.equations_.iloc[-1]["loss"], 1e-4)
        np.testing.assert_almost_equal(regressor.predict(X), y, decimal=1)

        # Tweak model selection:
        regressor.set_params(model_selection="best")
        self.assertEqual(regressor.get_params()["model_selection"], "best")
        self.assertTrue("None" not in regressor.__repr__())
        self.assertTrue(">>>>" in regressor.__repr__())

    def test_warm_start_set_at_init(self):
        # Smoke test for bug where warm_start=True is set at init
        y = self.X[:, 0]
        regressor = PySRRegressor(warm_start=True, max_evals=10)
        regressor.fit(self.X, y)

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
            early_stop_condition="stop_if(loss, complexity) = loss < 0.05 && complexity == 2",
        )
        # We expect in this case that the "best"
        # equation should be the right one:
        model.set_params(model_selection="best")
        # Also try without a temp equation file:
        model.set_params(temp_equation_file=False)
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
            denoise=True,
            nested_constraints={"/": {"+": 1, "-": 1}, "+": {"*": 4}},
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-3 && complexity == 7",
        )
        model.fit(X, y, Xresampled=Xresampled)
        self.assertNotIn("unused_feature", model.latex())
        self.assertIn("T", model.latex())
        self.assertIn("x", model.latex())
        self.assertLessEqual(model.get_best()["loss"], 1e-1)
        fn = model.get_best()["lambda_format"]
        X2 = pd.DataFrame(
            {
                "T": self.rstate.randn(100),
                "unused_feature": self.rstate.randn(100),
                "x": self.rstate.randn(100),
            }
        )
        self.assertLess(np.average((fn(X2) - true_fn(X2)) ** 2), 1e-1)
        self.assertLess(np.average((model.predict(X2) - true_fn(X2)) ** 2), 1e-1)

    def test_high_dim_selection_early_stop(self):
        X = pd.DataFrame({f"k{i}": self.rstate.randn(10000) for i in range(10)})
        Xresampled = pd.DataFrame({f"k{i}": self.rstate.randn(100) for i in range(10)})
        y = X["k7"] ** 2 + np.cos(X["k9"]) * 3

        model = PySRRegressor(
            unary_operators=["cos"],
            select_k_features=3,
            early_stop_condition=1e-4,  # Stop once most accurate equation is <1e-4 MSE
            maxsize=12,
            **self.default_test_kwargs,
        )
        model.set_params(model_selection="accuracy")
        model.fit(X, y, Xresampled=Xresampled)
        self.assertLess(np.average((model.predict(X) - y) ** 2), 1e-4)
        # Again, but with numpy arrays:
        model.fit(X.values, y.values, Xresampled=Xresampled.values)
        self.assertLess(np.average((model.predict(X.values) - y.values) ** 2), 1e-4)


class TestBest(unittest.TestCase):
    def setUp(self):
        self.rstate = np.random.RandomState(0)
        self.X = self.rstate.randn(10, 2)
        self.y = np.cos(self.X[:, 0]) ** 2
        self.model = PySRRegressor(
            progress=False,
            niterations=1,
            extra_sympy_mappings={},
            output_jax_format=False,
            model_selection="accuracy",
            equation_file="equation_file.csv",
        )
        equations = pd.DataFrame(
            {
                "equation": ["1.0", "cos(x0)", "square(cos(x0))"],
                "loss": [1.0, 0.1, 1e-5],
                "complexity": [1, 2, 3],
            }
        )

        # Set up internal parameters as if it had been fitted:
        self.model.equation_file_ = "equation_file.csv"
        self.model.nout_ = 1
        self.model.selection_mask_ = None
        self.model.feature_names_in_ = np.array(["x0", "x1"], dtype=object)
        equations["complexity loss equation".split(" ")].to_csv(
            "equation_file.csv.bkup", sep="|"
        )

        self.model.refresh()
        self.equations_ = self.model.equations_

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
        X = self.X
        y = self.y
        for f in [self.model.predict, self.equations_.iloc[-1]["lambda_format"]]:
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
        """Ensure that deprecation works as expected.

        This should give a warning, and sets the correct value.
        """
        with self.assertWarns(FutureWarning):
            model = PySRRegressor(fractionReplaced=0.2)
        # This is a deprecated parameter, so we should get a warning.

        # The correct value should be set:
        self.assertEqual(model.fraction_replaced, 0.2)

    def test_size_warning(self):
        """Ensure that a warning is given for a large input size."""
        model = PySRRegressor()
        X = np.random.randn(10001, 2)
        y = np.random.randn(10001)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(Exception) as context:
                model.fit(X, y)
            self.assertIn("more than 10,000", str(context.exception))

    def test_feature_warning(self):
        """Ensure that a warning is given for large number of features."""
        model = PySRRegressor()
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(Exception) as context:
                model.fit(X, y)
            self.assertIn("with 10 features or more", str(context.exception))

    def test_deterministic_warnings(self):
        """Ensure that warnings are given for determinism"""
        model = PySRRegressor(random_state=0)
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(Exception) as context:
                model.fit(X, y)
            self.assertIn("`deterministic`", str(context.exception))

    def test_deterministic_errors(self):
        """Setting deterministic without random_state should error"""
        model = PySRRegressor(deterministic=True)
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_pickle_with_temp_equation_file(self):
        """If we have a temporary equation file, unpickle the estimator."""
        model = PySRRegressor(
            populations=int(1 + DEFAULT_POPULATIONS / 5),
            temp_equation_file=True,
            procs=0,
            multithreading=False,
        )
        nout = 3
        X = np.random.randn(100, 2)
        y = np.random.randn(100, nout)
        model.fit(X, y)
        contents = model.equation_file_contents_.copy()

        y_predictions = model.predict(X)

        equation_file_base = model.equation_file_
        for i in range(1, nout + 1):
            assert not os.path.exists(str(equation_file_base) + f".out{i}.bkup")

        with tempfile.NamedTemporaryFile() as pickle_file:
            pkl.dump(model, pickle_file)
            pickle_file.seek(0)
            model2 = pkl.load(pickle_file)

        contents2 = model2.equation_file_contents_
        cols_to_check = ["equation", "loss", "complexity"]
        for frame1, frame2 in zip(contents, contents2):
            pd.testing.assert_frame_equal(frame1[cols_to_check], frame2[cols_to_check])

        y_predictions2 = model2.predict(X)
        np.testing.assert_array_equal(y_predictions, y_predictions2)

    def test_scikit_learn_compatibility(self):
        """Test PySRRegressor compatibility with scikit-learn."""
        model = PySRRegressor(
            niterations=int(1 + DEFAULT_NITERATIONS / 10),
            populations=int(1 + DEFAULT_POPULATIONS / 3),
            ncyclesperiteration=int(2 + DEFAULT_NCYCLES / 10),
            verbosity=0,
            progress=False,
            random_state=0,
            deterministic=True,  # Deterministic as tests require this.
            procs=0,
            multithreading=False,
            warm_start=False,
            temp_equation_file=True,
        )  # Return early.

        check_generator = check_estimator(model, generate_only=True)
        exception_messages = []
        for (_, check) in check_generator:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    check(model)
                print("Passed", check.func.__name__)
            except Exception:
                error_message = str(traceback.format_exc())
                exception_messages.append(
                    f"{check.func.__name__}:\n" + error_message + "\n"
                )
                print("Failed", check.func.__name__, "with:")
                # Add a leading tab to error message, which
                # might be multi-line:
                print("\n".join([(" " * 4) + row for row in error_message.split("\n")]))
        # If any checks failed don't let the test pass.
        self.assertEqual(len(exception_messages), 0)
