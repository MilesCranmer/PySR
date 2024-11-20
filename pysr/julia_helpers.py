"""Functions for initializing the Julia environment and installing deps."""

from typing import Any, Callable, Union, cast

import numpy as np
from juliacall import convert as jl_convert  # type: ignore
from numpy.typing import NDArray

from .deprecated import init_julia, install
from .julia_import import jl

jl_convert = cast(Callable[[Any, Any], Any], jl_convert)

jl.seval("using Serialization: Serialization")
jl.seval("using PythonCall: PythonCall")

Serialization = jl.Serialization
PythonCall = jl.PythonCall

jl.seval("using SymbolicRegression: plus, sub, mult, div, pow")


def _escape_filename(filename):
    """Turn a path into a string with correctly escaped backslashes."""
    str_repr = str(filename)
    str_repr = str_repr.replace("\\", "\\\\")
    return str_repr


def _load_cluster_manager(cluster_manager: str, mpi_flags: str):
    if cluster_manager == "mpi":
        jl.seval("using Distributed: addprocs")
        jl.seval("using MPIClusterManagers: MPIWorkerManager")

        return jl.seval(
            "(np; exeflags=``, kws...) -> "
            + "addprocs(MPIWorkerManager(np);"
            + ",".join(
                [
                    "exeflags=`$exeflags --project=$(Base.active_project())`",
                    f"mpiflags=`{mpi_flags}`",
                    "kws...",
                ]
            )
            + ")"
        )
    else:
        jl.seval(f"using ClusterManagers: addprocs_{cluster_manager}")
        return jl.seval(f"addprocs_{cluster_manager}")


def jl_array(x, dtype=None):
    if x is None:
        return None
    elif dtype is None:
        return jl_convert(jl.Array, x)
    else:
        return jl_convert(jl.Array[dtype], x)


def jl_is_function(f) -> bool:
    # We name it so we only compile it once
    is_function = jl.seval("__pysr_jl_is_function(op) = op isa Function")
    return cast(bool, is_function(f))


def jl_serialize(obj: Any) -> NDArray[np.uint8]:
    buf = jl.IOBuffer()
    Serialization.serialize(buf, obj)
    return np.array(jl.take_b(buf))


def jl_deserialize(s: Union[NDArray[np.uint8], None]):
    if s is None:
        return s
    buf = jl.IOBuffer()
    jl.write(buf, jl_array(s))
    jl.seekstart(buf)
    return Serialization.deserialize(buf)
