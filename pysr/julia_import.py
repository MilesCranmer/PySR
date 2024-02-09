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


def is_in_jupyter() -> bool:
    try:
        ipy = get_ipython().__class__.__name__  # type: ignore
        return ipy == "ZMQInteractiveShell"
    except NameError:
        return False


if is_in_jupyter():
    get_ipython().run_line_magic("load_ext", "julia.ipython")  # type: ignore


from juliacall import Main as jl  # type: ignore


# TODO: Overwrite this once PythonCall.jl is updated:
def seval(s: str):
    return jl.eval(jl.Meta.parseall(s))


jl.seval = seval
