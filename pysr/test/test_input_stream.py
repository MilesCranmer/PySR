"""Tests for input_stream default behavior."""

from pysr import PySRRegressor


def test_default_is_none():
    assert PySRRegressor().input_stream is None


def test_explicit_values():
    assert PySRRegressor(input_stream="stdin").input_stream == "stdin"
    assert PySRRegressor(input_stream="devnull").input_stream == "devnull"
