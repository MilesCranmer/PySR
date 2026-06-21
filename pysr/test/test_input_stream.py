"""Tests for input_stream default behavior."""

import unittest
from unittest import mock

import numpy as np

from pysr import PySRRegressor
from pysr.sr import ALREADY_RAN


class TestInputStream(unittest.TestCase):
    """Verify input_stream defaults and backend passthrough."""

    def test_default_is_none(self):
        """By default, input_stream should be None so the Julia backend picks the stream."""
        model = PySRRegressor()
        self.assertIsNone(model.input_stream)

    def test_explicit_stdin(self):
        model = PySRRegressor(input_stream="stdin")
        self.assertEqual(model.input_stream, "stdin")

    def test_explicit_devnull(self):
        model = PySRRegressor(input_stream="devnull")
        self.assertEqual(model.input_stream, "devnull")

    def _make_mock_jl(self):
        """Return a MagicMock that satisfies the Julia calls made before Options()."""
        m = mock.MagicMock()
        m.seval.return_value = mock.MagicMock()
        m.Dict.return_value = mock.MagicMock()
        m.Pair.return_value = mock.MagicMock()
        m.Symbol.return_value = mock.MagicMock()
        m.NamedTuple.return_value = mock.MagicMock()
        return m

    def _mocked_fit(self, model, X, y, *, capture_options):
        """Run model.fit with enough Julia infrastructure mocked to reach Options()."""
        mock_jl = self._make_mock_jl()
        sr = mock.MagicMock()
        sr.MutationWeights.return_value = mock.MagicMock()
        sr.Options.side_effect = capture_options
        sr.equation_search.return_value = (mock.MagicMock(), b"mock")
        sr.SearchUtilsModule.generate_run_id.return_value = "test-run-id"

        # Reset global so _run doesn't skip its first-run block.
        import pysr.sr as sr_module

        old_already_ran = sr_module.ALREADY_RAN
        sr_module.ALREADY_RAN = False

        try:
            with mock.patch("pysr.sr.jl", mock_jl):
                with mock.patch("pysr.sr.SymbolicRegression", sr):
                    with mock.patch("pysr.sr.jl_array", return_value=mock.MagicMock()):
                        with mock.patch("pysr.sr.jl_is_function", return_value=True):
                            with mock.patch(
                                "pysr.sr.jl_serialize", return_value=b"mock"
                            ):
                                with mock.patch("pysr.sr.load_required_packages"):
                                    with mock.patch("pysr.sr._load_cluster_manager"):
                                        model.fit(X, y)
        finally:
            sr_module.ALREADY_RAN = old_already_ran

    def test_default_omits_input_stream_from_options(self):
        """When input_stream is None, Options should not receive the kwarg."""
        captured_kwargs = {}

        def capture_options(**kwargs):
            captured_kwargs.update(kwargs)
            raise RuntimeError("stop_after_options")

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])

        model = PySRRegressor(
            niterations=0,
            max_evals=0,
            populations=1,
            ncycles_per_iteration=0,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            parallelism="serial",
        )

        with self.assertRaises(RuntimeError) as cm:
            self._mocked_fit(model, X, y, capture_options=capture_options)
        self.assertEqual(str(cm.exception), "stop_after_options")
        self.assertNotIn("input_stream", captured_kwargs)

    def test_explicit_stdin_passes_input_stream(self):
        """When input_stream is 'stdin', Options should receive the kwarg."""
        captured_kwargs = {}

        def capture_options(**kwargs):
            captured_kwargs.update(kwargs)
            raise RuntimeError("stop_after_options")

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])

        model = PySRRegressor(
            input_stream="stdin",
            niterations=0,
            max_evals=0,
            populations=1,
            ncycles_per_iteration=0,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            parallelism="serial",
        )

        with self.assertRaises(RuntimeError) as cm:
            self._mocked_fit(model, X, y, capture_options=capture_options)
        self.assertEqual(str(cm.exception), "stop_after_options")
        self.assertIn("input_stream", captured_kwargs)

    def test_explicit_devnull_passes_input_stream(self):
        """When input_stream is 'devnull', Options should receive the kwarg."""
        captured_kwargs = {}

        def capture_options(**kwargs):
            captured_kwargs.update(kwargs)
            raise RuntimeError("stop_after_options")

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])

        model = PySRRegressor(
            input_stream="devnull",
            niterations=0,
            max_evals=0,
            populations=1,
            ncycles_per_iteration=0,
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            parallelism="serial",
        )

        with self.assertRaises(RuntimeError) as cm:
            self._mocked_fit(model, X, y, capture_options=capture_options)
        self.assertEqual(str(cm.exception), "stop_after_options")
        self.assertIn("input_stream", captured_kwargs)


def runtests(just_tests=False):
    tests = [TestInputStream]
    if just_tests:
        return tests
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for test in tests:
        suite.addTests(loader.loadTestsFromTestCase(test))
    runner = unittest.TextTestRunner()
    return runner.run(suite)
