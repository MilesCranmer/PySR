"""Functions to create a sysimage for PySR."""

from pathlib import Path
import numpy as np

from . import PySRRegressor
from .julia_helpers import init_julia


def create_sysimage(sysimage_name="pysr.so", julia_project=None, quiet=False):
    """Create a PackageCompiler.jl sysimage from a simple PySR run."""

    Main = init_julia(julia_project=julia_project, quiet=quiet)
    cur_project_dir = Main.eval("dirname(Base.active_project())")
    sysimage_path = str(Path(cur_project_dir) / sysimage_name)
    from julia import PackageCompiler

    PackageCompiler.create_sysimage(["SymbolicRegression"], sysimage_path=sysimage_path)
