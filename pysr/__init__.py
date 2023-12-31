import sys
import warnings

if sys.version_info >= (3, 12, 0):
    warnings.warn(
        "PySR experiences occassional segfaults with Python 3.12. "
        + "Please use an earlier version of Python with PySR until this issue is resolved."
    )

from . import sklearn_monkeypatch
from .deprecated import best, best_callable, best_row, best_tex, pysr
from .export_jax import sympy2jax
from .export_torch import sympy2torch
from .feynman_problems import FeynmanProblem, Problem
from .julia_helpers import install
from .sr import PySRRegressor
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
