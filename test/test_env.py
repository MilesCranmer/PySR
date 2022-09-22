"""Contains tests for creating and initializing custom Julia projects"""

import unittest
from pysr import julia_helpers


class TestJuliaProject(unittest.TestCase):
    def test_custom_shared_env(self):
        """Test that we can use PySR in a custom shared env"""
        test_env_name = "@pysr_test_env"
        julia_helpers.install(julia_project=test_env_name, quiet=True)
        Main = julia_helpers.init_julia(julia_project=test_env_name)
        Main.eval("using SymbolicRegression")
        Main.eval("using Pkg")
        cur_project_dir = Main.eval("splitdir(dirname(Base.active_project()))[1]")
        potential_shared_project_dirs = Main.eval("Pkg.envdir.(DEPOT_PATH)")
        self.assertIn(cur_project_dir, potential_shared_project_dirs)
        # TODO: Delete the env at the end.
