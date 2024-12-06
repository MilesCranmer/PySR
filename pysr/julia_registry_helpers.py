"""Utilities for managing Julia registry preferences during package operations."""

import os
import warnings
from collections.abc import Callable


def backup_juliaregistrypref():
    """Backup and potentially modify Julia registry preference.

    Sets JULIA_PKG_SERVER_REGISTRY_PREFERENCE to 'eager' if not already set.
    Returns the original value for later restoration.
    """
    old_value = os.environ.get("JULIA_PKG_SERVER_REGISTRY_PREFERENCE", None)
    if old_value is None:
        warnings.warn(
            "Attempting to use the `eager` registry flavor of the Julia "
            "General registry from the Julia Pkg server.\n"
            "    If any errors are encountered, try setting the "
            "`JULIA_PKG_SERVER_REGISTRY_PREFERENCE` environment variable to `conservative`."
        )
        os.environ["JULIA_PKG_SERVER_REGISTRY_PREFERENCE"] = "eager"
    return old_value


def restore_juliaregistrypref(old_value: str | None):
    """Restore the original Julia registry preference value."""
    if old_value is None:
        del os.environ["JULIA_PKG_SERVER_REGISTRY_PREFERENCE"]
    else:
        os.environ["JULIA_PKG_SERVER_REGISTRY_PREFERENCE"] = old_value


def with_juliaregistrypref(f: Callable[..., None], *args):
    """Execute function with modified Julia registry preference.

    First tries with existing registry preference. If that fails with a Julia registry error,
    temporarily modifies the registry preference to 'eager'. Restores original preference after
    execution.

    Parameters
    ----------
    f : Callable[..., None]
        Function to execute. Should not return anything of importance.
    *args : Any
        Arguments to pass to the function.
    """
    try:
        f(*args)
        return
    except Exception as initial_error:
        # Check if this is a Julia registry error by looking at the error message
        error_str = str(initial_error)
        if (
            "JuliaError" not in error_str
            or "Unsatisfiable requirements detected" not in error_str
        ):
            raise initial_error

        old_value = os.environ.get("JULIA_PKG_SERVER_REGISTRY_PREFERENCE", None)
        if old_value == "eager":
            raise initial_error

        warnings.warn(
            "Initial Julia registry operation failed. Attempting to use the `eager` registry flavor of the Julia "
            "General registry from the Julia Pkg server (via the `JULIA_PKG_SERVER_REGISTRY_PREFERENCE` environment variable)."
        )
        os.environ["JULIA_PKG_SERVER_REGISTRY_PREFERENCE"] = "eager"
        try:
            f(*args)
        finally:
            if old_value is not None:
                os.environ["JULIA_PKG_SERVER_REGISTRY_PREFERENCE"] = old_value
            else:
                del os.environ["JULIA_PKG_SERVER_REGISTRY_PREFERENCE"]
