"""Test that running PySR with static libpython raises a warning."""

import warnings
import pysr

# Taken from https://stackoverflow.com/a/14463362/2689923
with warnings.catch_warnings(record=True) as warning_catcher:
    warnings.simplefilter("always")
    pysr.sr.init_julia()

    assert len(warning_catcher) == 1
    assert issubclass(warning_catcher[-1].category, UserWarning)
    assert "static" in str(warning_catcher[-1].message)
