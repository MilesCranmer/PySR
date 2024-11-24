import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from .export import add_export_formats
from .julia_helpers import jl_array
from .julia_import import SymbolicRegression, jl

if TYPE_CHECKING:
    from .sr import PySRRegressor


class AbstractExpressionOptions(ABC):
    """Abstract base class for expression types.

    This basically just holds the options for the expression type,
    as well as explains how to parse and evaluate them.

    All expression types must implement:

    1. julia_expression_type(): The actual expression type, returned as a Julia object.
        This will get stored as `expression_type` in `SymbolicRegression.Options`.
    2. julia_expression_options(): Method to create the expression options, returned as a Julia object.
        These will get stored as `expression_options` in `SymbolicRegression.Options`.
    3. load_from(): whether expressions are read from the hall of fame file, or loaded from Julia.

    You can also optionally implement create_exports(), which will be used to
    create the exports of the equations.
    """

    @abstractmethod
    def julia_expression_type(self) -> Any:
        """The expression type"""
        pass

    @abstractmethod
    def julia_expression_options(self) -> Any:
        """The expression options"""
        pass

    @abstractmethod
    def load_from(self) -> Literal["file", "julia"]:
        """If expressions are read from the hall of fame file, or loaded from Julia"""
        pass

    def create_exports(
        self,
        model: "PySRRegressor",
        equations: pd.DataFrame,
        search_output: Any,
    ) -> pd.DataFrame:
        return add_export_formats(
            equations,
            feature_names_in=model.feature_names_in_,
            selection_mask=model.selection_mask_,
            extra_sympy_mappings=model.extra_sympy_mappings,
            extra_torch_mappings=model.extra_torch_mappings,
            output_jax_format=model.output_jax_format,
            extra_jax_mappings=model.extra_jax_mappings,
            output_torch_format=model.output_torch_format,
        )


class ExpressionOptions(AbstractExpressionOptions):
    """Options for the regular Expression expression type"""

    def julia_expression_type(self):
        return SymbolicRegression.Expression

    def julia_expression_options(self):
        return jl.NamedTuple()

    def load_from(self):
        return "file"


class CallableJuliaExpression:
    def __init__(self, expression):
        self.expression = expression

    def __call__(self, X: np.ndarray):
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        raw_output = self.expression(jl_array(X.T))
        return np.array(raw_output).T


class TemplateExpressionOptions(AbstractExpressionOptions):
    """The structure of a template expression.

    This class allows you to specify how multiple sub-expressions should be combined
    in a structured way, with constraints on which variables each sub-expression can use.
    Pass this to PySRRegressor with the `expression_options` argument when you are using
    the `TemplateExpression` expression type.

    Parameters
    ----------
    function_symbols : list[str]
        List of symbols representing the inner expressions (e.g., ["f", "g"]).
        These will be used as keys in the template structure.
    combine : str
        Julia function string that defines how the sub-expressions are combined.
        Takes a NamedTuple of expressions and a tuple of data vectors.
        For example: "((; f, g), (x1, x2, x3)) -> f(x1, x2) + g(x3)^2"
        would constrain f to use x1,x2 and g to use x3.
    num_features : dict[str, int]
        Dictionary mapping function symbols to the number of features each can use.
        For example: {"f": 2, "g": 1} means f takes 2 inputs and g takes 1.
        If not provided, will be inferred from the combine function.

    Examples
    --------
    ```python
    # Create template that combines f(x1, x2) and g(x3):
    template_options = TemplateExpressionOptions(
        function_symbols=["f", "g"],
        combine="((; f, g), (x1, x2, x3)) -> sin(f(x1, x2)) + g(x3)^2",
    )

    # Use in PySRRegressor:
    model = PySRRegressor(
        expression_options=template_options
    )
    ```
    """

    def __init__(
        self,
        function_symbols: List[str],
        combine: str,
        num_features: Optional[Dict[str, int]] = None,
    ):
        self.function_symbols = function_symbols
        self.combine = combine
        self.num_features = num_features

    def julia_expression_type(self):
        return SymbolicRegression.TemplateExpression

    def julia_expression_options(self):
        f_combine = jl.seval(self.combine)
        creator = jl.seval(
            """
        function _pysr_create_template_structure(
            @nospecialize(function_symbols::AbstractVector),
            @nospecialize(combine::Function),
            @nospecialize(num_features::Union{Nothing,AbstractDict})
        )
            tuple_symbol = (map(Symbol, function_symbols)..., )
            num_features = if num_features === nothing
                nothing
            else
                (; num_features...)
            end
            return SymbolicRegression.TemplateStructure{tuple_symbol}(combine, num_features)
        end
        """
        )
        structure = creator(self.function_symbols, f_combine, self.num_features)
        return jl.seval("NamedTuple{(:structure,)}")((structure,))

    def load_from(self):
        return "julia"

    def create_exports(
        self,
        model: "PySRRegressor",
        equations: pd.DataFrame,
        search_output: Any,
    ) -> pd.DataFrame:
        assert search_output is not None

        equations = copy.deepcopy(equations)

        (_, out_hof) = search_output
        expressions = []
        callables = []
        scores = []

        lastMSE = None
        lastComplexity = 0

        for _, row in equations.iterrows():
            curComplexity = row["complexity"]
            curMSE = row["loss"]
            expression = out_hof.members[curComplexity - 1].tree
            expressions.append(expression)
            callables.append(CallableJuliaExpression(expression))

            if lastMSE is None:
                cur_score = 0.0
            else:
                if curMSE > 0.0:
                    # TODO Move this to more obvious function/file.
                    cur_score = -np.log(curMSE / lastMSE) / (
                        curComplexity - lastComplexity
                    )
                else:
                    cur_score = np.inf
            scores.append(cur_score)

        equations["julia_expression"] = expressions
        equations["lambda_format"] = callables
        equations["score"] = np.array(scores)
        return equations
