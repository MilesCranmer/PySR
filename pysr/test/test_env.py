"""Contains tests for creating and initializing custom Julia projects."""

import unittest
import os
from tempfile import TemporaryDirectory

from .. import julia_helpers


class TestJuliaProject(unittest.TestCase):
    """Various tests for working with Julia projects."""

    def test_custom_shared_env(self):
        """Test that we can use PySR in a custom shared env."""
        with TemporaryDirectory() as tmpdir:
            # Create a temp depot to store julia packages (and our custom env)
            Main = julia_helpers.init_julia()

            # Set up env:
            if "JULIA_DEPOT_PATH" not in os.environ:
                old_env = None
                os.environ["JULIA_DEPOT_PATH"] = tmpdir
            else:
                old_env = os.environ["JULIA_DEPOT_PATH"]
                os.environ[
                    "JULIA_DEPOT_PATH"
                ] = f"{tmpdir}:{os.environ['JULIA_DEPOT_PATH']}"
            Main.eval(
                f'pushfirst!(DEPOT_PATH, "{julia_helpers._escape_filename(tmpdir)}")'
            )
            test_env_name = "@pysr_test_env"
            julia_helpers.install(julia_project=test_env_name)
            Main = julia_helpers.init_julia(julia_project=test_env_name)

            # Try to use env:
            Main.eval("using SymbolicRegression")
            Main.eval("using Pkg")

            # Assert we actually loaded it:
            cur_project_dir = Main.eval("splitdir(dirname(Base.active_project()))[1]")
            potential_shared_project_dirs = Main.eval("Pkg.envdir(DEPOT_PATH[1])")
            self.assertEqual(cur_project_dir, potential_shared_project_dirs)

            # Clean up:
            Main.eval("pop!(DEPOT_PATH)")
            if old_env is None:
                del os.environ["JULIA_DEPOT_PATH"]
            else:
                os.environ["JULIA_DEPOT_PATH"] = old_env


def runtests():
    """Run all tests in test_env.py."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestJuliaProject))
    runner = unittest.TextTestRunner()
    return runner.run(suite)
