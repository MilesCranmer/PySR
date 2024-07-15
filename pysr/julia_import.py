import os
import sys
import warnings
from types import ModuleType
from typing import cast

import juliapkg

# Check if JuliaCall is already loaded, and if so, warn the user
# about the relevant environment variables. If not loaded,
# set up sensible defaults.
if "juliacall" in sys.modules:
    warnings.warn(
        "juliacall module already imported. "
        "Make sure that you have set the environment variable `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` to avoid segfaults. "
        "Also note that PySR will not be able to configure `PYTHON_JULIACALL_THREADS` or `PYTHON_JULIACALL_OPTLEVEL` for you."
    )
else:
    # Required to avoid segfaults (https://juliapy.github.io/PythonCall.jl/dev/faq/)
    if os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes") != "yes":
        warnings.warn(
            "PYTHON_JULIACALL_HANDLE_SIGNALS environment variable is set to something other than 'yes' or ''. "
            + "You will experience segfaults if running with multithreading."
        )

    if os.environ.get("PYTHON_JULIACALL_THREADS", "auto") != "auto":
        warnings.warn(
            "PYTHON_JULIACALL_THREADS environment variable is set to something other than 'auto', "
            "so PySR was not able to set it. You may wish to set it to `'auto'` for full use "
            "of your CPU."
        )

    # TODO: Remove these when juliapkg lets you specify this
    for k, default in (
        ("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes"),
        ("PYTHON_JULIACALL_THREADS", "auto"),
        ("PYTHON_JULIACALL_OPTLEVEL", "3"),
    ):
        os.environ[k] = os.environ.get(k, default)

juliapkg.require_julia("~1.6.7, ~1.7, ~1.8, ~1.9, =1.10.0, ^1.10.3")
juliapkg.add(
    "SymbolicRegression",
    "8254be44-1295-4e6a-a16d-46603ac705cb",
    version="=0.24.5",
)
juliapkg.add("Serialization", "9e88b42a-f829-5b0c-bbe9-9e923198166b", version="1")


autoload_extensions = os.environ.get("PYSR_AUTOLOAD_EXTENSIONS")
if autoload_extensions is not None:
    # Deprecated; so just pass to juliacall
    os.environ["PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION"] = autoload_extensions

from juliacall import Main as jl  # type: ignore

jl = cast(ModuleType, jl)


jl_version = (jl.VERSION.major, jl.VERSION.minor, jl.VERSION.patch)

jl.seval("using SymbolicRegression")
SymbolicRegression = jl.SymbolicRegression

jl.seval("using Pkg: Pkg")
Pkg = jl.Pkg
