import os
import platform
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from pysr import PySRRegressor
from pysr.julia_import import jl_version

from .params import DEFAULT_NITERATIONS, DEFAULT_POPULATIONS


class TestStartup(unittest.TestCase):
    """Various tests related to starting up PySR."""

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

    def test_warm_start_from_file(self):
        """Test that we can warm start in another process."""
        if platform.system() == "Windows":
            self.skipTest("Warm start test incompatible with Windows")

        with tempfile.TemporaryDirectory() as tmpdirname:
            model = PySRRegressor(
                **self.default_test_kwargs,
                unary_operators=["cos"],
            )
            model.warm_start = True
            model.temp_equation_file = False
            model.equation_file = Path(tmpdirname) / "equations.csv"
            model.deterministic = True
            model.multithreading = False
            model.random_state = 0
            model.procs = 0
            model.early_stop_condition = 1e-10

            rstate = np.random.RandomState(0)
            X = rstate.randn(100, 2)
            y = np.cos(X[:, 0]) ** 2
            model.fit(X, y)

            best_loss = model.equations_.iloc[-1]["loss"]

            # Save X and y to a file:
            X_file = Path(tmpdirname) / "X.npy"
            y_file = Path(tmpdirname) / "y.npy"
            np.save(X_file, X)
            np.save(y_file, y)
            # Now, create a new process and warm start from the file:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent(
                        f"""
                        from pysr import PySRRegressor
                        import numpy as np

                        X = np.load("{X_file}")
                        y = np.load("{y_file}")

                        print("Loading model from file")
                        model = PySRRegressor.from_file("{model.equation_file}")

                        assert model.julia_state_ is not None

                        # Reset saved equations; should be loaded from state!
                        model.equations_ = None
                        model.equation_file_contents_ = None

                        model.warm_start = True
                        model.niterations = 0
                        model.max_evals = 0
                        model.ncycles_per_iteration = 0

                        model.fit(X, y)

                        best_loss = model.equations_.iloc[-1]["loss"]

                        assert best_loss <= {best_loss}
                    """
                    ),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ,
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("Loading model from file", result.stdout.decode())
            self.assertIn("Started!", result.stderr.decode())

    def test_bad_startup_options(self):
        warning_tests = [
            dict(
                code='import os; os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "no"; import pysr',
                msg="PYTHON_JULIACALL_HANDLE_SIGNALS environment variable is set",
            ),
            dict(
                code='import os; os.environ["PYTHON_JULIACALL_THREADS"] = "1"; import pysr',
                msg="PYTHON_JULIACALL_THREADS environment variable is set",
            ),
            dict(
                code="import juliacall; import pysr",
                msg="juliacall module already imported.",
            ),
        ]
        for warning_test in warning_tests:
            result = subprocess.run(
                [sys.executable, "-c", warning_test["code"]],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ,
            )
            self.assertIn(warning_test["msg"], result.stderr.decode())

    def test_notebook(self):
        if jl_version < (1, 9, 0):
            self.skipTest("Julia version too old")
        if platform.system() == "Windows":
            self.skipTest("Notebook test incompatible with Windows")

        notebook_file = Path(__file__).parent / "test_nb.ipynb"
        sanitize_file = Path(__file__).parent / "nb_sanitize.cfg"

        if not (notebook_file.exists() and sanitize_file.exists()):
            self.skipTest("Files not available for testing")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--nbval",
                str(notebook_file),
                "--nbval-sanitize-with",
                str(sanitize_file),
            ],
            env=os.environ,
        )
        self.assertEqual(result.returncode, 0)


def runtests(just_tests=False):
    tests = [TestStartup]
    if just_tests:
        return tests
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for test in tests:
        suite.addTests(loader.loadTestsFromTestCase(test))
    runner = unittest.TextTestRunner()
    return runner.run(suite)
