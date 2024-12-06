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

    Temporarily modifies the registry preference to 'eager', falling back to
    'conservative' if network errors occur. Restores original preference after
    execution.

    Parameters
    ----------
    f : Callable[..., None]
        Function to execute. Should not return anything of importance.
    *args : Any
        Arguments to pass to the function.
    """
    old_value = backup_juliaregistrypref()
    try:
        f(*args)
    except Exception as e:
        error_msg = (
            "ERROR: Encountered a network error.\n"
            "    Are you behind a firewall, or are there network restrictions that would "
            "prevent access to certain websites or domains?\n"
            "    Try setting the `JULIA_PKG_SERVER_REGISTRY_PREFERENCE` environment "
            "variable to `conservative`."
        )
        if old_value is not None:
            warnings.warn(error_msg)
            restore_juliaregistrypref(old_value)
            raise e
        else:
            os.environ["JULIA_PKG_SERVER_REGISTRY_PREFERENCE"] = "conservative"
            try:
                f(args)
            except Exception as e:
                warnings.warn(error_msg)
                restore_juliaregistrypref(old_value)
                raise e
