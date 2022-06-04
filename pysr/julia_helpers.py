"""Functions for initializing the Julia environment and installing deps."""
import warnings
from pathlib import Path
import os

from .version import __version__, __symbolic_regression_jl_version__


def install(julia_project=None, quiet=False):  # pragma: no cover
    """
    Install PyCall.jl and all required dependencies for SymbolicRegression.jl.

    Also updates the local Julia registry.
    """
    import julia

    julia.install(quiet=quiet)

    julia_project, is_shared = _get_julia_project(julia_project)

    Main = init_julia()
    Main.eval("using Pkg")

    io = "devnull" if quiet else "stderr"
    io_arg = f"io={io}" if is_julia_version_greater_eq(Main, "1.6") else ""

    # Can't pass IO to Julia call as it evaluates to PyObject, so just directly
    # use Main.eval:
    Main.eval(
        f'Pkg.activate("{_escape_filename(julia_project)}", shared = Bool({int(is_shared)}), {io_arg})'
    )
    if is_shared:
        # Install SymbolicRegression.jl:
        _add_sr_to_julia_project(Main, io_arg)

    Main.eval(f"Pkg.instantiate({io_arg})")
    Main.eval(f"Pkg.precompile({io_arg})")
    if not quiet:
        warnings.warn(
            "It is recommended to restart Python after installing PySR's dependencies,"
            " so that the Julia environment is properly initialized."
        )


def import_error_string(julia_project=None):
    s = """
    Required dependencies are not installed or built.  Run the following code in the Python REPL:

        >>> import pysr
        >>> pysr.install()
    """

    if julia_project is not None:
        s += f"""
        Tried to activate project {julia_project} but failed."""

    return s


def _get_julia_project(julia_project):
    if julia_project is None:
        is_shared = True
        julia_project = f"pysr-{__version__}"
    else:
        is_shared = False
        julia_project = Path(julia_project)
    return julia_project, is_shared


def is_julia_version_greater_eq(Main, version="1.6"):
    """Check if Julia version is greater than specified version."""
    return Main.eval(f'VERSION >= v"{version}"')


def init_julia():
    """Initialize julia binary, turning off compiled modules if needed."""
    from julia.core import JuliaInfo, UnsupportedPythonError

    try:
        info = JuliaInfo.load(julia="julia")
    except FileNotFoundError:
        env_path = os.environ["PATH"]
        raise FileNotFoundError(
            f"Julia is not installed in your PATH. Please install Julia and add it to your PATH.\n\nCurrent PATH: {env_path}",
        )

    if not info.is_pycall_built():
        raise ImportError(import_error_string())

    Main = None
    try:
        from julia import Main as _Main

        Main = _Main
    except UnsupportedPythonError:
        # Static python binary, so we turn off pre-compiled modules.
        from julia.core import Julia

        jl = Julia(compiled_modules=False)
        from julia import Main as _Main

        Main = _Main

    return Main


def _add_sr_to_julia_project(Main, io_arg):
    Main.sr_spec = Main.PackageSpec(
        name="SymbolicRegression",
        url="https://github.com/MilesCranmer/SymbolicRegression.jl",
        rev="v" + __symbolic_regression_jl_version__,
    )
    Main.eval(f"Pkg.add(sr_spec, {io_arg})")
    Main.clustermanagers_spec = Main.PackageSpec(
        name="ClusterManagers",
        url="https://github.com/JuliaParallel/ClusterManagers.jl",
        rev="14e7302f068794099344d5d93f71979aaf4fbeb3",
    )
    Main.eval(f"Pkg.add(clustermanagers_spec, {io_arg})")


def _escape_filename(filename):
    """Turns a file into a string representation with correctly escaped backslashes"""
    str_repr = str(filename)
    str_repr = str_repr.replace("\\", "\\\\")
    return str_repr
