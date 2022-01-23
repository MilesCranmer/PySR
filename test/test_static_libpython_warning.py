"""Test that running PySR with static libpython raises a warning.

Note: This test will ONLY pass with statically-linked python binaries, such
as provided by conda. It will not pass on other versions of python, and that is
okay."""

import unittest
import warnings
import pysr

# Taken from https://stackoverflow.com/a/14463362/2689923
class TestLibpythonWarning(unittest.TestCase):
    def test_warning(self):
        with warnings.catch_warnings(record=True) as warning_catcher:
            warnings.simplefilter("always")
            pysr.sr.init_julia()

            self.assertEqual(len(warning_catcher), 1)
            self.assertTrue(issubclass(warning_catcher[-1].category, UserWarning))
            self.assertIn("static", str(warning_catcher[-1].message))
