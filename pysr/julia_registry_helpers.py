"""Utilities for managing Julia registry preferences during package operations."""

import os
import warnings
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def try_with_registry_fallback(f: Callable[..., T], *args, **kwargs) -> T:
    """Execute function with modified Julia registry preference.

    First tries with existing registry preference. If that fails with a Julia registry error,
    temporarily modifies the registry preference to 'eager'. Restores original preference after
    execution.
    """
    try:
        return f(*args, **kwargs)
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
            return f(*args, **kwargs)
        finally:
            if old_value is not None:
                os.environ["JULIA_PKG_SERVER_REGISTRY_PREFERENCE"] = old_value
            else:
                del os.environ["JULIA_PKG_SERVER_REGISTRY_PREFERENCE"]
