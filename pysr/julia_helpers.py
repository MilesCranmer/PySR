"""Functions for initializing the Julia environment and installing deps."""
import os
import sys
import warnings

if "juliacall" in sys.modules:
    warnings.warn(
        "juliacall module already imported. Make sure that you have set `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` to avoid segfaults."
    )

# Required to avoid segfaults (https://juliapy.github.io/PythonCall.jl/dev/faq/)
if os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes") != "yes":
    warnings.warn(
        "PYTHON_JULIACALL_HANDLE_SIGNALS environment variable is set to something other than 'yes' or ''. "
        + "You will experience segfaults if running with multithreading."
    )

if os.environ.get("JULIA_NUM_THREADS", "auto") != "auto":
    warnings.warn(
        "JULIA_NUM_THREADS environment variable is set to something other than 'auto', "
        "so PySR was not able to set it. You may wish to set it to `'auto'` for full use "
        "of your CPU."
    )

# TODO: Remove these when juliapkg lets you specify this
for k, default in (
    ("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes"),
    ("JULIA_NUM_THREADS", "auto"),
    ("JULIA_OPTIMIZE", "3"),
):
    os.environ[k] = os.environ.get(k, default)


import juliapkg
from juliacall import Main as jl
from juliacall import convert as jl_convert

jl.seval("using Serialization: Serialization")
jl.seval("using PythonCall: PythonCall")

juliainfo = None
julia_initialized = False
julia_kwargs_at_initialization = None
julia_activated_env = None


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
