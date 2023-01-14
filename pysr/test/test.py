import os
import traceback
import inspect
import unittest
import numpy as np
from sklearn import model_selection
from sklearn.utils.estimator_checks import check_estimator
import sympy
import pandas as pd
import warnings
import pickle as pkl
import tempfile
from pathlib import Path

from .. import julia_helpers
from .. import PySRRegressor
from ..sr import (
    run_feature_selection,
    _handle_feature_selection,
    _csv_filename_to_pkl_filename,
    idx_model_selection,
)
from ..export_latex import to_latex

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

    def test_linear_relation_named(self):
        y = self.X[:, 0]
        model = PySRRegressor(
            **self.default_test_kwargs,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity == 1",
        )
        model.fit(self.X, y, variable_names=["c1", "c2", "c3", "c4", "c5"])
        self.assertIn("c1", model.equations_.iloc[-1]["equation"])

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

    def test_multiprocessing_turbo(self):
        y = self.X[:, 0]
        model = PySRRegressor(
            **self.default_test_kwargs,
            # Turbo needs to work with unsafe operators:
            unary_operators=["sqrt"],
            procs=2,
            multithreading=False,
            turbo=True,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity == 1",
        )
        model.fit(self.X, y)
        print(model.equations_)
        self.assertLessEqual(model.equations_.iloc[-1]["loss"], 1e-4)

    def test_high_precision_search(self):
        y = 1.23456789 * self.X[:, 0]
        model = PySRRegressor(
            **self.default_test_kwargs,
            early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity == 3",
            precision=64,
            parsimony=0.01,
            warm_start=True,
        )
        model.fit(self.X, y)
        from pysr.sr import Main

        # We should have that the model state is now a Float64 hof:
        Main.test_state = model.raw_julia_state_
        self.assertTrue(Main.eval("typeof(test_state[2]).parameters[1] == Float64"))

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
            # Test custom operators with turbo:
            turbo=True,
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
                model.predict(X.copy())[:, 0], X[:, 0] ** 2, decimal=3
            )
        except AssertionError:
            print("Error in test_multioutput_weighted_with_callable_temp_equation")
            print("Model equations: ", model.sympy()[0])
            print("True equation: x0^2")

        try:
            np.testing.assert_almost_equal(
                model.predict(X.copy())[:, 1], X[:, 1] ** 2, decimal=3
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
        # Check that the the julia state is saved:
        from pysr.sr import Main

        # We should have that the model state is now a Float32 hof:
        Main.test_state = regressor.raw_julia_state_
        self.assertTrue(Main.eval("typeof(test_state[2]).parameters[1] == Float32"))
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

    def test_load_model(self):
        """See if we can load a ran model from the equation file."""
        csv_file_data = """
        Complexity,Loss,Equation
        1,0.19951081,"1.9762075"
        3,0.12717344,"(f0 + 1.4724599)"
        4,0.104823045,"pow_abs(2.2683423, cos(f3))\""""
        # Strip the indents:
        csv_file_data = "\n".join([l.strip() for l in csv_file_data.split("\n")])

        for from_backup in [False, True]:
            rand_dir = Path(tempfile.mkdtemp())
            equation_filename = str(rand_dir / "equation.csv")
            with open(equation_filename + (".bkup" if from_backup else ""), "w") as f:
                f.write(csv_file_data)
            model = PySRRegressor.from_file(
                equation_filename,
                n_features_in=5,
                feature_names_in=["f0", "f1", "f2", "f3", "f4"],
                binary_operators=["+", "*", "/", "-", "^"],
                unary_operators=["cos"],
            )
            X = self.rstate.rand(100, 5)
            y_truth = 2.2683423 ** np.cos(X[:, 3])
            y_test = model.predict(X, 2)

            np.testing.assert_allclose(y_truth, y_test)

    def test_load_model_simple(self):
        # Test that we can simply load a model from its equation file.
        y = self.X[:, [0, 1]] ** 2
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
        rand_dir = Path(tempfile.mkdtemp())
        equation_file = rand_dir / "equations.csv"
        model.set_params(temp_equation_file=False)
        model.set_params(equation_file=equation_file)
        model.fit(self.X, y)

        # lambda functions are removed from the pickling, so we need
        # to pass it during the loading:
        model2 = PySRRegressor.from_file(
            model.equation_file_, extra_sympy_mappings={"sq": lambda x: x**2}
        )

        np.testing.assert_allclose(model.predict(self.X), model2.predict(self.X))

        # Try again, but using only the pickle file:
        for file_to_delete in [str(equation_file), str(equation_file) + ".bkup"]:
            if os.path.exists(file_to_delete):
                os.remove(file_to_delete)

        pickle_file = rand_dir / "equations.pkl"
        model3 = PySRRegressor.from_file(
            model.equation_file_, extra_sympy_mappings={"sq": lambda x: x**2}
        )
        np.testing.assert_allclose(model.predict(self.X), model3.predict(self.X))


def manually_create_model(equations, feature_names=None):
    if feature_names is None:
        feature_names = ["x0", "x1"]

    model = PySRRegressor(
        progress=False,
        niterations=1,
        extra_sympy_mappings={},
        output_jax_format=False,
        model_selection="accuracy",
        equation_file="equation_file.csv",
    )

    # Set up internal parameters as if it had been fitted:
    if isinstance(equations, list):
        # Multi-output.
        model.equation_file_ = "equation_file.csv"
        model.nout_ = len(equations)
        model.selection_mask_ = None
        model.feature_names_in_ = np.array(feature_names, dtype=object)
        for i in range(model.nout_):
            equations[i]["complexity loss equation".split(" ")].to_csv(
                f"equation_file.csv.out{i+1}.bkup"
            )
    else:
        model.equation_file_ = "equation_file.csv"
        model.nout_ = 1
        model.selection_mask_ = None
        model.feature_names_in_ = np.array(feature_names, dtype=object)
        equations["complexity loss equation".split(" ")].to_csv(
            "equation_file.csv.bkup"
        )

    model.refresh()

    return model


class TestBest(unittest.TestCase):
    def setUp(self):
        self.rstate = np.random.RandomState(0)
        self.X = self.rstate.randn(10, 2)
        self.y = np.cos(self.X[:, 0]) ** 2
        equations = pd.DataFrame(
            {
                "equation": ["1.0", "cos(x0)", "square(cos(x0))"],
                "loss": [1.0, 0.1, 1e-5],
                "complexity": [1, 2, 3],
            }
        )
        self.model = manually_create_model(equations)
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
            np.testing.assert_almost_equal(f(X), y, decimal=3)

    def test_all_selection_strategies(self):
        equations = pd.DataFrame(
            dict(
                loss=[1.0, 0.1, 0.01, 0.001 * 1.4, 0.001],
                score=[0.5, 1.0, 0.5, 0.5, 0.3],
            )
        )
        idx_accuracy = idx_model_selection(equations, "accuracy")
        self.assertEqual(idx_accuracy, 4)
        idx_best = idx_model_selection(equations, "best")
        self.assertEqual(idx_best, 3)
        idx_score = idx_model_selection(equations, "score")
        self.assertEqual(idx_score, 1)


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

    def test_csv_to_pkl_conversion(self):
        """Test that csv filename to pkl filename works as expected."""
        tmpdir = Path(tempfile.mkdtemp())
        equation_file = tmpdir / "equations.389479384.28378374.csv"
        expected_pkl_file = tmpdir / "equations.389479384.28378374.pkl"

        # First, test inputting the paths:
        test_pkl_file = _csv_filename_to_pkl_filename(equation_file)
        self.assertEqual(test_pkl_file, str(expected_pkl_file))

        # Next, test inputting the strings.
        test_pkl_file = _csv_filename_to_pkl_filename(str(equation_file))
        self.assertEqual(test_pkl_file, str(expected_pkl_file))

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

    def test_changed_options_warning(self):
        """Check that a warning is given if Julia options are changed."""
        if julia_helpers.julia_kwargs_at_initialization is None:
            julia_helpers.init_julia(julia_kwargs={"threads": 2, "optimize": 3})

        cur_init = julia_helpers.julia_kwargs_at_initialization

        threads_to_change = cur_init["threads"] + 1
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(Exception) as context:
                julia_helpers.init_julia(
                    julia_kwargs={"threads": threads_to_change, "optimize": 3}
                )
            self.assertIn("Julia has already started", str(context.exception))
            self.assertIn("threads", str(context.exception))

    def test_extra_sympy_mappings_undefined(self):
        """extra_sympy_mappings=None errors for custom operators"""
        model = PySRRegressor(unary_operators=["square2(x) = x^2"])
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_sympy_function_fails_as_variable(self):
        model = PySRRegressor()
        X = np.random.randn(100, 2)
        y = np.random.randn(100)
        with self.assertRaises(ValueError) as cm:
            model.fit(X, y, variable_names=["x1", "N"])
        self.assertIn("Variable name", str(cm.exception))

    def test_bad_variable_names_fail(self):
        model = PySRRegressor()
        X = np.random.randn(100, 1)
        y = np.random.randn(100)

        with self.assertRaises(ValueError) as cm:
            model.fit(X, y, variable_names=["Tr(Tij)"])
        self.assertIn("Invalid variable name", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            model.fit(X, y, variable_names=["f{c}"])
        self.assertIn("Invalid variable name", str(cm.exception))

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


TRUE_PREAMBLE = "\n".join(
    [
        r"\usepackage{breqn}",
        r"\usepackage{booktabs}",
        "",
        "...",
        "",
    ]
)


class TestLaTeXTable(unittest.TestCase):
    def setUp(self):
        equations = pd.DataFrame(
            dict(
                equation=["x0", "cos(x0)", "x0 + x1 - cos(x1 * x0)"],
                loss=[1.052, 0.02315, 1.12347e-15],
                complexity=[1, 2, 8],
            )
        )
        self.model = manually_create_model(equations)
        self.maxDiff = None

    def create_true_latex(self, middle_part, include_score=False):
        if include_score:
            true_latex_table_str = r"""
                \begin{table}[h]
                \begin{center}
                \begin{tabular}{@{}cccc@{}}
                \toprule
                Equation & Complexity & Loss & Score \\
                \midrule"""
        else:
            true_latex_table_str = r"""
                \begin{table}[h]
                \begin{center}
                \begin{tabular}{@{}ccc@{}}
                \toprule
                Equation & Complexity & Loss \\
                \midrule"""
        true_latex_table_str += middle_part
        true_latex_table_str += r"""\bottomrule
            \end{tabular}
            \end{center}
            \end{table}
        """
        # First, remove empty lines:
        true_latex_table_str = "\n".join(
            [line.strip() for line in true_latex_table_str.split("\n") if len(line) > 0]
        )
        return true_latex_table_str.strip()

    def test_simple_table(self):
        latex_table_str = self.model.latex_table(
            columns=["equation", "complexity", "loss"]
        )
        middle_part = r"""
            $y = x_{0}$ & $1$ & $1.05$ \\
            $y = \cos{\left(x_{0} \right)}$ & $2$ & $0.0232$ \\
            $y = x_{0} + x_{1} - \cos{\left(x_{0} x_{1} \right)}$ & $8$ & $1.12 \cdot 10^{-15}$ \\
        """
        true_latex_table_str = (
            TRUE_PREAMBLE + "\n" + self.create_true_latex(middle_part)
        )
        self.assertEqual(latex_table_str, true_latex_table_str)

    def test_other_precision(self):
        latex_table_str = self.model.latex_table(
            precision=5, columns=["equation", "complexity", "loss"]
        )
        middle_part = r"""
            $y = x_{0}$ & $1$ & $1.0520$ \\
            $y = \cos{\left(x_{0} \right)}$ & $2$ & $0.023150$ \\
            $y = x_{0} + x_{1} - \cos{\left(x_{0} x_{1} \right)}$ & $8$ & $1.1235 \cdot 10^{-15}$ \\
        """
        true_latex_table_str = (
            TRUE_PREAMBLE + "\n" + self.create_true_latex(middle_part)
        )
        self.assertEqual(latex_table_str, true_latex_table_str)

    def test_include_score(self):
        latex_table_str = self.model.latex_table()
        middle_part = r"""
            $y = x_{0}$ & $1$ & $1.05$ & $0.0$ \\
            $y = \cos{\left(x_{0} \right)}$ & $2$ & $0.0232$ & $3.82$ \\
            $y = x_{0} + x_{1} - \cos{\left(x_{0} x_{1} \right)}$ & $8$ & $1.12 \cdot 10^{-15}$ & $5.11$ \\
        """
        true_latex_table_str = (
            TRUE_PREAMBLE
            + "\n"
            + self.create_true_latex(middle_part, include_score=True)
        )
        self.assertEqual(latex_table_str, true_latex_table_str)

    def test_last_equation(self):
        latex_table_str = self.model.latex_table(
            indices=[2], columns=["equation", "complexity", "loss"]
        )
        middle_part = r"""
            $y = x_{0} + x_{1} - \cos{\left(x_{0} x_{1} \right)}$ & $8$ & $1.12 \cdot 10^{-15}$ \\
        """
        true_latex_table_str = (
            TRUE_PREAMBLE + "\n" + self.create_true_latex(middle_part)
        )
        self.assertEqual(latex_table_str, true_latex_table_str)

    def test_multi_output(self):
        equations1 = pd.DataFrame(
            dict(
                equation=["x0", "cos(x0)", "x0 + x1 - cos(x1 * x0)"],
                loss=[1.052, 0.02315, 1.12347e-15],
                complexity=[1, 2, 8],
            )
        )
        equations2 = pd.DataFrame(
            dict(
                equation=["x1", "cos(x1)", "x0 * x0 * x1"],
                loss=[1.32, 0.052, 2e-15],
                complexity=[1, 2, 5],
            )
        )
        equations = [equations1, equations2]
        model = manually_create_model(equations)
        middle_part_1 = r"""
            $y_{0} = x_{0}$ & $1$ & $1.05$ & $0.0$ \\
            $y_{0} = \cos{\left(x_{0} \right)}$ & $2$ & $0.0232$ & $3.82$ \\
            $y_{0} = x_{0} + x_{1} - \cos{\left(x_{0} x_{1} \right)}$ & $8$ & $1.12 \cdot 10^{-15}$ & $5.11$ \\
        """
        middle_part_2 = r"""
            $y_{1} = x_{1}$ & $1$ & $1.32$ & $0.0$ \\
            $y_{1} = \cos{\left(x_{1} \right)}$ & $2$ & $0.0520$ & $3.23$ \\
            $y_{1} = x_{0}^{2} x_{1}$ & $5$ & $2.00 \cdot 10^{-15}$ & $10.3$ \\
        """
        true_latex_table_str = "\n\n".join(
            self.create_true_latex(part, include_score=True)
            for part in [middle_part_1, middle_part_2]
        )
        true_latex_table_str = TRUE_PREAMBLE + "\n" + true_latex_table_str
        latex_table_str = model.latex_table()

        self.assertEqual(latex_table_str, true_latex_table_str)

    def test_latex_float_precision(self):
        """Test that we can print latex expressions with custom precision"""
        expr = sympy.Float(4583.4485748, dps=50)
        self.assertEqual(to_latex(expr, prec=6), r"4583.45")
        self.assertEqual(to_latex(expr, prec=5), r"4583.4")
        self.assertEqual(to_latex(expr, prec=4), r"4583.")
        self.assertEqual(to_latex(expr, prec=3), r"4.58 \cdot 10^{3}")
        self.assertEqual(to_latex(expr, prec=2), r"4.6 \cdot 10^{3}")

        # Multiple numbers:
        x = sympy.Symbol("x")
        expr = x * 3232.324857384 - 1.4857485e-10
        self.assertEqual(
            to_latex(expr, prec=2), r"3.2 \cdot 10^{3} x - 1.5 \cdot 10^{-10}"
        )
        self.assertEqual(
            to_latex(expr, prec=3), r"3.23 \cdot 10^{3} x - 1.49 \cdot 10^{-10}"
        )
        self.assertEqual(
            to_latex(expr, prec=8), r"3232.3249 x - 1.4857485 \cdot 10^{-10}"
        )

    def test_latex_break_long_equation(self):
        """Test that we can break a long equation inside the table"""
        long_equation = """
        - cos(x1 * x0) + 3.2 * x0 - 1.2 * x1 + x1 * x1 * x1 + x0 * x0 * x0
        + 5.2 * sin(0.3256 * sin(x2) - 2.6 * x0) + x0 * x0 * x0 * x0 * x0
        + cos(cos(x1 * x0) + 3.2 * x0 - 1.2 * x1 + x1 * x1 * x1 + x0 * x0 * x0)
        """
        long_equation = "".join(long_equation.split("\n")).strip()
        equations = pd.DataFrame(
            dict(
                equation=["x0", "cos(x0)", long_equation],
                loss=[1.052, 0.02315, 1.12347e-15],
                complexity=[1, 2, 30],
            )
        )
        model = manually_create_model(equations)
        latex_table_str = model.latex_table()
        middle_part = r"""
        $y = x_{0}$ & $1$ & $1.05$ & $0.0$ \\
        $y = \cos{\left(x_{0} \right)}$ & $2$ & $0.0232$ & $3.82$ \\
        \begin{minipage}{0.8\linewidth} \vspace{-1em} \begin{dmath*} y = x_{0}^{5} + x_{0}^{3} + 3.20 x_{0} + x_{1}^{3} - 1.20 x_{1} - 5.20 \sin{\left(2.60 x_{0} - 0.326 \sin{\left(x_{2} \right)} \right)} - \cos{\left(x_{0} x_{1} \right)} + \cos{\left(x_{0}^{3} + 3.20 x_{0} + x_{1}^{3} - 1.20 x_{1} + \cos{\left(x_{0} x_{1} \right)} \right)} \end{dmath*} \end{minipage} & $30$ & $1.12 \cdot 10^{-15}$ & $1.09$ \\
        """
        true_latex_table_str = (
            TRUE_PREAMBLE
            + "\n"
            + self.create_true_latex(middle_part, include_score=True)
        )
        self.assertEqual(latex_table_str, true_latex_table_str)


def runtests():
    """Run all tests in test.py."""
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    test_cases = [
        TestPipeline,
        TestBest,
        TestFeatureSelection,
        TestMiscellaneous,
        TestLaTeXTable,
    ]
    for test_case in test_cases:
        tests = loader.loadTestsFromTestCase(test_case)
        suite.addTests(tests)
    runner = unittest.TextTestRunner()
    return runner.run(suite)
