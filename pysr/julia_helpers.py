"""Functions for initializing the Julia environment and installing deps."""
from juliacall import convert as jl_convert  # type: ignore

from .julia_import import jl

jl.seval("using Serialization: Serialization")
jl.seval("using PythonCall: PythonCall")


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


def jl_deserialize_s(s):
    if s is None:
        return s
    buf = jl.IOBuffer()
    jl.write(buf, jl_array(s))
    jl.seekstart(buf)
    return jl.Serialization.deserialize(buf)
