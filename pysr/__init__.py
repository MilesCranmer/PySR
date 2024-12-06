import os

if os.environ.get("PYSR_USE_BEARTYPE", "0") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()

# This must be imported as early as possible to prevent
# library linking issues caused by numpy/pytorch/etc. importing
# old libraries:
from .julia_import import jl, SymbolicRegression  # isort:skip

from . import sklearn_monkeypatch
from .deprecated import best, best_callable, best_row, best_tex, install, pysr
from .export_jax import sympy2jax
from .export_torch import sympy2torch
from .expression_specs import (
    AbstractExpressionSpec,
    ExpressionSpec,
    ParametricExpressionSpec,
    TemplateExpressionSpec,
)
from .julia_extensions import load_all_packages
from .logger_specs import AbstractLoggerSpec, TensorBoardLoggerSpec
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
    "AbstractExpressionSpec",
    "ExpressionSpec",
    "TemplateExpressionSpec",
    "ParametricExpressionSpec",
    "AbstractLoggerSpec",
    "TensorBoardLoggerSpec",
    "best",
    "best_callable",
    "best_row",
    "best_tex",
    "pysr",
    "__version__",
]
