import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, NewType, TypeAlias

import numpy as np
import pandas as pd

from .export import add_export_formats
from .julia_helpers import jl_array
from .julia_import import AnyValue, SymbolicRegression, jl

# For type checking purposes
if TYPE_CHECKING:
    from .sr import PySRRegressor  # pragma: no cover

    PySRRegressor: TypeAlias = PySRRegressor  # pragma: no cover
else:
    PySRRegressor = NewType("PySRRegressor", Any)


class AbstractExpressionSpec(ABC):
    """Abstract base class describing expression types.

    This basically just holds the options for the expression type,
    as well as explains how to parse and evaluate them.

    All expression types must implement:

    1. julia_expression_type(): The actual expression type, returned as a Julia object.
        This will get stored as `expression_type` in `SymbolicRegression.Options`.
    2. julia_expression_options(): Method to create the expression options, returned as a Julia object.
        These will get stored as `expression_options` in `SymbolicRegression.Options`.
    3. create_exports(), which will be used to create the exports of the equations, such as
        the executable format, the SymPy format, etc.

    It may also optionally implement:

    - supports_sympy, supports_torch, supports_jax, supports_latex: Whether this expression type supports the corresponding export format.
    """

    @abstractmethod
    def julia_expression_type(self) -> AnyValue:
        """The expression type"""
        pass  # pragma: no cover

    @abstractmethod
    def julia_expression_options(self) -> AnyValue:
        """The expression options"""
        pass  # pragma: no cover

    @abstractmethod
    def create_exports(
        self,
        model: PySRRegressor,
        equations: pd.DataFrame,
        search_output,
    ) -> pd.DataFrame:
        """Create additional columns in the equations dataframe."""
        pass  # pragma: no cover

    @property
    def evaluates_in_julia(self) -> bool:
        return False

    @property
    def supports_sympy(self) -> bool:
        return False

    @property
    def supports_torch(self) -> bool:
        return False

    @property
    def supports_jax(self) -> bool:
        return False

    @property
    def supports_latex(self) -> bool:
        return False


class ExpressionSpec(AbstractExpressionSpec):
    """The default expression specification, with no special behavior."""

    def julia_expression_type(self):
        return SymbolicRegression.Expression

    def julia_expression_options(self):
        return jl.NamedTuple()

    def create_exports(
        self,
        model: PySRRegressor,
        equations: pd.DataFrame,
        search_output,
    ):
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

    @property
    def supports_sympy(self):
        return True

    @property
    def supports_torch(self):
        return True

    @property
    def supports_jax(self):
        return True

    @property
    def supports_latex(self):
        return True


class TemplateExpressionSpec(AbstractExpressionSpec):
    """Spec for templated expressions.

    This class allows you to specify how multiple sub-expressions should be combined
    in a structured way, with constraints on which variables each sub-expression can use.
    Pass this to PySRRegressor with the `expression_spec` argument when you are using
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
    expression_spec = TemplateExpressionSpec(
        function_symbols=["f", "g"],
        combine="((; f, g), (x1, x2, x3)) -> sin(f(x1, x2)) + g(x3)^2",
    )

    # Use in PySRRegressor:
    model = PySRRegressor(
        expression_spec=expression_spec
    )
    ```
    """

    def __init__(
        self,
        function_symbols: list[str],
        combine: str,
        num_features: dict[str, int] | None = None,
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
            structure = SymbolicRegression.TemplateStructure{tuple_symbol}(combine, num_features)
            return (; structure)
        end
        """
        )
        return creator(self.function_symbols, f_combine, self.num_features)

    @property
    def evaluates_in_julia(self):
        return True

    def create_exports(
        self,
        model: PySRRegressor,
        equations: pd.DataFrame,
        search_output,
    ) -> pd.DataFrame:
        # We try to load the raw julia state from a saved binary stream
        # if not provided.
        search_output = search_output or model.julia_state_
        return _search_output_to_callable_expressions(equations, search_output)


class ParametricExpressionSpec(AbstractExpressionSpec):
    """Spec for parametric expressions that vary by category.

    This class allows you to specify expressions with parameters that vary across different
    categories in your dataset. The expression structure remains the same, but parameters
    are optimized separately for each category.

    Parameters
    ----------
    max_parameters : int
        Maximum number of parameters that can appear in the expression. Each parameter
        will take on different values for each category in the data.

    Examples
    --------
    For example, if we want to allow for a model with up to 2 parameters (each category
    can have a different value for these parameters), we can use:

    ```python
    model = PySRRegressor(
        expression_spec=ParametricExpressionSpec(max_parameters=2),
        binary_operators=["+", "*"],
        unary_operators=["sin"]
    )
    model.fit(X, y, category=category)
    ```
    """

    def __init__(self, max_parameters: int):
        self.max_parameters = max_parameters

    def julia_expression_type(self):
        return SymbolicRegression.ParametricExpression

    def julia_expression_options(self):
        return jl.seval("NamedTuple{(:max_parameters,)}")((self.max_parameters,))

    @property
    def evaluates_in_julia(self):
        return True

    def create_exports(
        self,
        model: PySRRegressor,
        equations: pd.DataFrame,
        search_output,
    ):
        search_output = search_output or model.julia_state_
        return _search_output_to_callable_expressions(equations, search_output)


class CallableJuliaExpression:
    def __init__(self, expression):
        self.expression = expression

    def __call__(self, X: np.ndarray, *args):
        raw_output = self.expression(jl_array(X.T), *args)
        return np.array(raw_output).T


def _search_output_to_callable_expressions(
    equations: pd.DataFrame, search_output
) -> pd.DataFrame:
    equations = copy.deepcopy(equations)
    (_, out_hof) = search_output
    expressions = []
    callables = []

    for _, row in equations.iterrows():
        curComplexity = row["complexity"]
        expression = out_hof.members[curComplexity - 1].tree
        expressions.append(expression)
        callables.append(CallableJuliaExpression(expression))

    df = pd.DataFrame(
        {"julia_expression": expressions, "lambda_format": callables},
        index=equations.index,
    )
    return df
