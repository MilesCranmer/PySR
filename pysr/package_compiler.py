"""Functions to create a sysimage for PySR."""

import os
from pathlib import Path
import warnings

import numpy as np
from julia.api import JuliaError

from .version import __symbolic_regression_jl_version__
from .julia_helpers import (
    init_julia,
    _julia_version_assertion,
    _set_julia_project_env,
    _get_io_arg,
    _process_julia_project,
    _import_error,
)


def _add_sr_to_julia_project(Main, io_arg):
    Main.eval("using Pkg")
    Main.sr_spec = Main.PackageSpec(
        name="SymbolicRegression",
        url="https://github.com/MilesCranmer/SymbolicRegression.jl",
        rev="v" + __symbolic_regression_jl_version__,
    )
    Main.clustermanagers_spec = Main.PackageSpec(
        name="ClusterManagers",
        rev="v0.4.2",
    )
    Main.packagecompiler_spec = Main.PackageSpec(
        name="PackageCompiler",
        rev="v2.1.0",
    )
    Main.eval(
        "Pkg.add(["
        + ", ".join(["sr_spec", "clustermanagers_spec", "packagecompiler_spec"])
        + f"], {io_arg})"
    )


def _update_julia_project(Main, is_shared, io_arg):
    try:
        if is_shared:
            _add_sr_to_julia_project(Main, io_arg)
        Main.eval("using Pkg")
        Main.eval(f"Pkg.resolve({io_arg})")
    except (JuliaError, RuntimeError) as e:
        raise ImportError(_import_error()) from e


def install(
    julia_project=None, quiet=False, precompile=None, compile=False
):  # pragma: no cover
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

    julia.install(quiet=quiet)
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

    if compile:
        create_sysimage(julia_project=julia_project, quiet=quiet)

    if not quiet:
        warnings.warn(
            "It is recommended to restart Python after installing PySR's dependencies,"
            " so that the Julia environment is properly initialized."
        )


def create_sysimage(sysimage_name="pysr.so", julia_project=None, quiet=False):
    """Create a PackageCompiler.jl sysimage for SymbolicRegression.jl."""
    Main = init_julia(julia_project=julia_project, quiet=quiet)
    cur_project_dir = Main.eval("dirname(Base.active_project())")
    sysimage_path = str(Path(cur_project_dir) / sysimage_name)
    from julia import PackageCompiler

    PackageCompiler.create_sysimage(["SymbolicRegression"], sysimage_path=sysimage_path)
