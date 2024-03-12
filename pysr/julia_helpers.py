"""Functions for initializing the Julia environment and installing deps."""

import numpy as np
from juliacall import convert as jl_convert  # type: ignore

from .deprecated import init_julia, install
from .julia_import import jl

jl.seval("using Serialization: Serialization")
jl.seval("using PythonCall: PythonCall")

Serialization = jl.Serialization
PythonCall = jl.PythonCall

jl.seval("using SymbolicRegression: plus, sub, mult, div, pow")


def _escape_filename(filename: str) -> str:
    """Turn a path into a string with correctly escaped backslashes."""
    return filename.replace("\\", "\\\\")


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
