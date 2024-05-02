import juliapkg

juliapkg.require_julia("~1.6.7, ~1.7, ~1.8, ~1.9, =1.10.0, ^1.10.3")
juliapkg.add(
    "SymbolicRegression", "8254be44-1295-4e6a-a16d-46603ac705cb", version="=0.24.4"
)
juliapkg.add("Serialization", "9e88b42a-f829-5b0c-bbe9-9e923198166b", version="1")

# This must be imported as early as possible to prevent
# library linking issues caused by numpy/pytorch/etc. importing
# old libraries:
from .julia_import import jl, SymbolicRegression  # isort:skip

from . import sklearn_monkeypatch
from .deprecated import best, best_callable, best_row, best_tex, install, pysr
from .export_jax import sympy2jax
from .export_torch import sympy2torch
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
    "PySRRegressor",
    "best",
    "best_callable",
    "best_row",
    "best_tex",
    "pysr",
    "__version__",
]
