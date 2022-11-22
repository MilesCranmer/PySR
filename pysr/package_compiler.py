"""Functions to create a sysimage for PySR."""

from pathlib import Path
import numpy as np

from . import PySRRegressor
from .julia_helpers import init_julia


def create_sysimage():
    """Create a PackageCompiler.jl sysimage from a simple PySR run."""
    sysimage_name = "pysr.so"

    # Example dataset:
    X = 2 * np.random.randn(100, 5)
    y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5

    model = PySRRegressor(
        unary_operators=["cos"],
        procs=8,
        niterations=2,
        populations=4,
        verbosity=0,
        progress=False,
    )
    model.fit(X, y)

    from julia import Main

    cur_project_dir = Main.eval("dirname(Base.active_project())")
    from julia import PackageCompiler

    PackageCompiler.create_sysimage(
        ["SymbolicRegression"],
        sysimage_path=str(Path(cur_project_dir) / sysimage_name),
    )
