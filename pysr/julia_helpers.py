"""Functions for initializing the Julia environment and installing deps."""
import os
import subprocess
import sys
import warnings
from pathlib import Path

from julia.api import JuliaError

from .version import __symbolic_regression_jl_version__, __version__

juliainfo = None
julia_initialized = False
julia_kwargs_at_initialization = None
julia_activated_env = None


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
            ["julia", "-e using Pkg; print(Pkg.envdir())"],
            capture_output=True,
            env=os.environ,
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


def _get_io_arg(quiet):
    io = "devnull" if quiet else "stderr"
    io_arg = f"io={io}" if is_julia_version_greater_eq(version=(1, 6, 0)) else ""
    return io_arg


def install(julia_project=None, quiet=False, precompile=None):  # pragma: no cover
    """
    Install PyCall.jl and all required dependencies for SymbolicRegression.jl.

    Also updates the local Julia registry.
    """
    import julia

    _julia_version_assertion()
    # Set JULIA_PROJECT so that we install in the pysr environment
    processed_julia_project, is_shared = _process_julia_project(julia_project)
    _set_julia_project_env(processed_julia_project, is_shared)

    if precompile == False:
        os.environ["JULIA_PKG_PRECOMPILE_AUTO"] = "0"

    try:
        julia.install(quiet=quiet)
    except julia.tools.PyCallInstallError:
        # Attempt to reset PyCall.jl's build:
        subprocess.run(
            [
                "julia",
                "-e",
                f'ENV["PYTHON"] = "{sys.executable}"; import Pkg; Pkg.build("PyCall")',
            ],
        )
        # Try installing again:
        try:
            julia.install(quiet=quiet)
        except julia.tools.PyCallInstallError:
            warnings.warn(
                "PyCall.jl failed to install on second attempt. "
                + "Please consult the GitHub issue "
                + "https://github.com/MilesCranmer/PySR/issues/257 "
                + "for advice on fixing this."
            )

    Main, init_log = init_julia(julia_project, quiet=quiet, return_aux=True)
    io_arg = _get_io_arg(quiet)

    if precompile is None:
        precompile = init_log["compiled_modules"]

    if not precompile:
        Main.eval('ENV["JULIA_PKG_PRECOMPILE_AUTO"] = 0')

    if is_shared:
        # Install SymbolicRegression.jl:
        _add_sr_to_julia_project(Main, io_arg)

    Main.eval("using Pkg")
    Main.eval(f"Pkg.instantiate({io_arg})")

    if precompile:
        Main.eval(f"Pkg.precompile({io_arg})")

    if not quiet:
        warnings.warn(
            "It is recommended to restart Python after installing PySR's dependencies,"
            " so that the Julia environment is properly initialized."
        )


def _import_error():
    return """
    Required dependencies are not installed or built.  Run the following command in your terminal:
        python3 -m pysr install
    """


def _process_julia_project(julia_project):
    if julia_project is None:
        is_shared = True
        processed_julia_project = f"pysr-{__version__}"
    elif julia_project[0] == "@":
        is_shared = True
        processed_julia_project = julia_project[1:]
    else:
        is_shared = False
        processed_julia_project = Path(julia_project)
    return processed_julia_project, is_shared


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


def init_julia(julia_project=None, quiet=False, julia_kwargs=None, return_aux=False):
    """Initialize julia binary, turning off compiled modules if needed."""
    global julia_initialized
    global julia_kwargs_at_initialization
    global julia_activated_env

    if not julia_initialized:
        _check_for_conflicting_libraries()

    if julia_kwargs is None:
        julia_kwargs = {"optimize": 3}

    from julia.core import JuliaInfo, UnsupportedPythonError

    _julia_version_assertion()
    processed_julia_project, is_shared = _process_julia_project(julia_project)
    _set_julia_project_env(processed_julia_project, is_shared)

    try:
        info = JuliaInfo.load(julia="julia")
    except FileNotFoundError:
        env_path = os.environ["PATH"]
        raise FileNotFoundError(
            f"Julia is not installed in your PATH. Please install Julia and add it to your PATH.\n\nCurrent PATH: {env_path}",
        )

    if not info.is_pycall_built():
        raise ImportError(_import_error())

    from julia.core import Julia

    try:
        Julia(**julia_kwargs)
    except UnsupportedPythonError:
        # Static python binary, so we turn off pre-compiled modules.
        julia_kwargs = {**julia_kwargs, "compiled_modules": False}
        Julia(**julia_kwargs)
        warnings.warn(
            "Your system's Python library is static (e.g., conda), so precompilation will be turned off. For a dynamic library, try using `pyenv` and installing with `--enable-shared`: https://github.com/pyenv/pyenv/blob/master/plugins/python-build/README.md#building-with---enable-shared."
        )

    using_compiled_modules = (not "compiled_modules" in julia_kwargs) or julia_kwargs[
        "compiled_modules"
    ]

    from julia import Main as _Main

    Main = _Main

    if julia_activated_env is None:
        julia_activated_env = processed_julia_project

    if julia_initialized and julia_kwargs_at_initialization is not None:
        # Check if the kwargs are the same as the previous initialization
        init_set = set(julia_kwargs_at_initialization.items())
        new_set = set(julia_kwargs.items())
        set_diff = new_set - init_set
        # Remove the `compiled_modules` key, since it is not a user-specified kwarg:
        set_diff = {k: v for k, v in set_diff if k != "compiled_modules"}
        if len(set_diff) > 0:
            warnings.warn(
                "Julia has already started. The new Julia options "
                + str(set_diff)
                + " will be ignored."
            )

    if julia_initialized and julia_activated_env != processed_julia_project:
        Main.eval("using Pkg")

        io_arg = _get_io_arg(quiet)
        # Can't pass IO to Julia call as it evaluates to PyObject, so just directly
        # use Main.eval:
        Main.eval(
            f'Pkg.activate("{_escape_filename(processed_julia_project)}",'
            f"shared = Bool({int(is_shared)}), "
            f"{io_arg})"
        )

        julia_activated_env = processed_julia_project

    if not julia_initialized:
        julia_kwargs_at_initialization = julia_kwargs

    julia_initialized = True
    if return_aux:
        return Main, {"compiled_modules": using_compiled_modules}
    return Main


def _add_sr_to_julia_project(Main, io_arg):
    Main.eval("using Pkg")
    Main.eval("Pkg.Registry.update()")
    Main.sr_spec = Main.PackageSpec(
        name="SymbolicRegression",
        url="https://github.com/MilesCranmer/SymbolicRegression.jl",
        rev="v" + __symbolic_regression_jl_version__,
    )
    Main.clustermanagers_spec = Main.PackageSpec(
        name="ClusterManagers",
        version="0.4",
    )
    Main.eval(f"Pkg.add([sr_spec, clustermanagers_spec], {io_arg})")


def _escape_filename(filename):
    """Turn a path into a string with correctly escaped backslashes."""
    str_repr = str(filename)
    str_repr = str_repr.replace("\\", "\\\\")
    return str_repr


def _julia_version_assertion():
    if not is_julia_version_greater_eq(version=(1, 6, 0)):
        raise NotImplementedError(
            "PySR requires Julia 1.6.0 or greater. "
            "Please update your Julia installation."
        )


def _backend_version_assertion(Main):
    try:
        backend_version = Main.eval("string(SymbolicRegression.PACKAGE_VERSION)")
        expected_backend_version = __symbolic_regression_jl_version__
        if backend_version != expected_backend_version:  # pragma: no cover
            warnings.warn(
                f"PySR backend (SymbolicRegression.jl) version {backend_version} "
                f"does not match expected version {expected_backend_version}. "
                "Things may break. "
                "Please update your PySR installation with "
                "`python3 -m pysr install`."
            )
    except JuliaError:  # pragma: no cover
        warnings.warn(
            "You seem to have an outdated version of SymbolicRegression.jl. "
            "Things may break. "
            "Please update your PySR installation with "
            "`python3 -m pysr install`."
        )


def _load_cluster_manager(Main, cluster_manager):
    Main.eval(f"import ClusterManagers: addprocs_{cluster_manager}")
    return Main.eval(f"addprocs_{cluster_manager}")


def _update_julia_project(Main, is_shared, io_arg):
    try:
        if is_shared:
            _add_sr_to_julia_project(Main, io_arg)
        Main.eval("using Pkg")
        Main.eval(f"Pkg.resolve({io_arg})")
    except (JuliaError, RuntimeError) as e:
        raise ImportError(_import_error()) from e


def _load_backend(Main):
    try:
        # Load namespace, so that various internal operators work:
        Main.eval("using SymbolicRegression")
    except (JuliaError, RuntimeError) as e:
        raise ImportError(_import_error()) from e

    _backend_version_assertion(Main)

    # Load Julia package SymbolicRegression.jl
    from julia import SymbolicRegression

    return SymbolicRegression
