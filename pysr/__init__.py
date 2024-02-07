from . import sklearn_monkeypatch
from .deprecated import best, best_callable, best_row, best_tex, pysr
from .export_jax import sympy2jax
from .export_torch import sympy2torch
from .feynman_problems import FeynmanProblem, Problem
from .sr import PySRRegressor, jl

# This file is created by setuptools_scm during the build process:
from .version import __version__

__all__ = [
    "sklearn_monkeypatch",
    "sympy2jax",
    "sympy2torch",
    "FeynmanProblem",
    "Problem",
    "install",
    "PySRRegressor",
    "best",
    "best_callable",
    "best_row",
    "best_tex",
    "pysr",
    "__version__",
]
