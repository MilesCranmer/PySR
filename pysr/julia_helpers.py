"""Functions for initializing the Julia environment and installing deps."""
import warnings

import numpy as np
from juliacall import convert as jl_convert  # type: ignore

from .julia_import import jl

jl.seval("using Serialization: Serialization")
jl.seval("using PythonCall: PythonCall")

Serialization = jl.Serialization
PythonCall = jl.PythonCall


def install(*args, **kwargs):
    del args, kwargs
    warnings.warn(
        "The `install` function has been removed. "
        "PySR now uses the `juliacall` package to install its dependencies automatically at import time. "
    )


def _escape_filename(filename):
    """Turn a path into a string with correctly escaped backslashes."""
    str_repr = str(filename)
    str_repr = str_repr.replace("\\", "\\\\")
    return str_repr


def _load_cluster_manager(cluster_manager):
    jl.seval(f"using ClusterManagers: addprocs_{cluster_manager}")
    return jl.seval(f"addprocs_{cluster_manager}")


def jl_array(x):
    if x is None:
        return None
    return jl_convert(jl.Array, x)


def jl_serialize(obj):
    buf = jl.IOBuffer()
    Serialization.serialize(buf, obj)
    return np.array(jl.take_b(buf))


def jl_deserialize(s):
    if s is None:
        return s
    buf = jl.IOBuffer()
    jl.write(buf, jl_array(s))
    jl.seekstart(buf)
    return Serialization.deserialize(buf)
