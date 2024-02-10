import os
import sys
import warnings

if "juliacall" in sys.modules:
    warnings.warn(
        "juliacall module already imported. "
        "Make sure that you have set the environment variable `PYTHON_JULIACALL_HANDLE_SIGNALS=yes` to avoid segfaults. "
        "Also note that PySR will not be able to configure `JULIA_NUM_THREADS` or `JULIA_OPTIMIZE` for you."
    )
else:
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

try:
    get_ipython = sys.modules["IPython"].get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")

    print("Detected Jupyter notebook. Loading juliacall extension.")

    # TODO: Turn this off if juliacall does this automatically
    get_ipython().run_line_magic("load_ext", "juliacall")
except Exception:
    pass


from juliacall import Main as jl  # type: ignore


# TODO: Overwrite this once PythonCall.jl is updated:
def seval(s: str):
    return jl.eval(jl.Meta.parseall(s))


jl.seval = seval
