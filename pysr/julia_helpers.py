"""Functions for initializing the Julia environment and installing deps."""
import sys
import subprocess
import warnings
from pathlib import Path
import os

from .version import __version__, __symbolic_regression_jl_version__

juliainfo = None
julia_initialized = False


def _load_juliainfo():
    """Execute julia.core.JuliaInfo.load(), and store as juliainfo."""
    global juliainfo

    if juliainfo is None:
        from julia.core import JuliaInfo

        try:
            juliainfo = JuliaInfo.load(julia="julia")
        except FileNotFoundError:
            env_path = os.environ["PATH"]
            raise FileNotFoundError(
                f"Julia is not installed in your PATH. Please install Julia and add it to your PATH.\n\nCurrent PATH: {env_path}",
            )

    return juliainfo


def _get_julia_env_dir():
    # Have to manually get env dir:
    try:
        julia_env_dir_str = subprocess.run(
            ["julia", "-e using Pkg; print(Pkg.envdir())"], capture_output=True
        ).stdout.decode()
    except FileNotFoundError:
        env_path = os.environ["PATH"]
        raise FileNotFoundError(
            f"Julia is not installed in your PATH. Please install Julia and add it to your PATH.\n\nCurrent PATH: {env_path}",
        )
    return Path(julia_env_dir_str)


def _set_julia_project_env(julia_project, is_shared):
    if is_shared:
        if is_julia_version_greater_eq(version=(1, 7, 0)):
            os.environ["JULIA_PROJECT"] = "@" + str(julia_project)
        else:
            julia_env_dir = _get_julia_env_dir()
            os.environ["JULIA_PROJECT"] = str(julia_env_dir / julia_project)
    else:
        os.environ["JULIA_PROJECT"] = str(julia_project)


def install(julia_project=None, quiet=False):  # pragma: no cover
    """
    Install PyCall.jl and all required dependencies for SymbolicRegression.jl.

    Also updates the local Julia registry.
    """
    import julia

    # Set JULIA_PROJECT so that we install in the pysr environment
    julia_project, is_shared = _process_julia_project(julia_project)
    _set_julia_project_env(julia_project, is_shared)

    julia.install(quiet=quiet)

    if is_shared:
        # is_shared is only true if the julia_project arg was None
        # See _process_julia_project
        Main = init_julia(None)
    else:
        Main = init_julia(julia_project)

    Main.eval("using Pkg")

    io = "devnull" if quiet else "stderr"
    io_arg = f"io={io}" if is_julia_version_greater_eq(version=(1, 6, 0)) else ""

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


def _import_error_string(julia_project=None):
    s = """
    Required dependencies are not installed or built.  Run the following code in the Python REPL:

        >>> import pysr
        >>> pysr.install()
    """

    if julia_project is not None:
        s += f"""
        Tried to activate project {julia_project} but failed."""

    return s


def _process_julia_project(julia_project):
    if julia_project is None:
        is_shared = True
        julia_project = f"pysr-{__version__}"
    else:
        is_shared = False
        julia_project = Path(julia_project)
    return julia_project, is_shared


def is_julia_version_greater_eq(juliainfo=None, version=(1, 6, 0)):
    """Check if Julia version is greater than specified version."""
    if juliainfo is None:
        juliainfo = _load_juliainfo()
    current_version = (
        juliainfo.version_major,
        juliainfo.version_minor,
        juliainfo.version_patch,
    )
    return current_version >= version


def _check_for_conflicting_libraries():  # pragma: no cover
    """Check whether there are conflicting modules, and display warnings."""
    # See https://github.com/pytorch/pytorch/issues/78829: importing
    # pytorch before running `pysr.fit` causes a segfault.
    torch_is_loaded = "torch" in sys.modules
    if torch_is_loaded:
        warnings.warn(
            "`torch` was loaded before the Julia instance started. "
            "This may cause a segfault when running `PySRRegressor.fit`. "
            "To avoid this, please run `pysr.julia_helpers.init_julia()` *before* "
            "importing `torch`. "
            "For updates, see https://github.com/pytorch/pytorch/issues/78829"
        )


def init_julia(julia_project=None):
    """Initialize julia binary, turning off compiled modules if needed."""
    global julia_initialized

    if not julia_initialized:
        _check_for_conflicting_libraries()

    from julia.core import JuliaInfo, UnsupportedPythonError

    julia_project, is_shared = _process_julia_project(julia_project)
    _set_julia_project_env(julia_project, is_shared)

    try:
        info = JuliaInfo.load(julia="julia")
    except FileNotFoundError:
        env_path = os.environ["PATH"]
        raise FileNotFoundError(
            f"Julia is not installed in your PATH. Please install Julia and add it to your PATH.\n\nCurrent PATH: {env_path}",
        )

    if not info.is_pycall_built():
        raise ImportError(_import_error_string())

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

    julia_initialized = True
    return Main


def _add_sr_to_julia_project(Main, io_arg):
    Main.sr_spec = Main.PackageSpec(
        name="SymbolicRegression",
        url="https://github.com/MilesCranmer/SymbolicRegression.jl",
        rev="v" + __symbolic_regression_jl_version__,
    )
    Main.clustermanagers_spec = Main.PackageSpec(
        name="ClusterManagers",
        url="https://github.com/JuliaParallel/ClusterManagers.jl",
        rev="14e7302f068794099344d5d93f71979aaf4fbeb3",
    )
    Main.eval(f"Pkg.add([sr_spec, clustermanagers_spec], {io_arg})")


def _escape_filename(filename):
    """Turn a path into a string with correctly escaped backslashes."""
    str_repr = str(filename)
    str_repr = str_repr.replace("\\", "\\\\")
    return str_repr
