# This must be imported as early as possible to prevent
# library linking issues caused by numpy/pytorch/etc. importing
# old libraries:
from .julia_import import jl, SymbolicRegression  # isort:skip

from . import sklearn_monkeypatch
from .deprecated import best, best_callable, best_row, best_tex, install, pysr
from .export_jax import sympy2jax
from .export_torch import sympy2torch
from .julia_extensions import load_all_packages
from .regressor_sequence import PySRSequenceRegressor
from .sr import PySRRegressor

# This file is created by setuptools_scm during the build process:
from .version import __version__

__all__ = [
    "jl",
    "SymbolicRegression",
    "sklearn_monkeypatch",
    "sympy2jax",
    "sympy2torch",
    "install",
    "load_all_packages",
    "PySRRegressor",
    "PySRSequenceRegressor",
    "best",
    "best_callable",
    "best_row",
    "best_tex",
    "pysr",
    "__version__",
]
