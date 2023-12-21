from numbers import Number
from typing import List, Optional

from .sr import PySRRegressor


class Expression:
    """A wrapper around `SymbolicRegression.Node`"""

    def __init__(
        self,
        equation,
        *,
        model: PySRRegressor = None,
        options=None,
        variable_names: Optional[List[str]] = None,
    ):
        super().__init__()
        # exactly one of model and options is None:
        assert (model is None) != (
            options is None
        ), "Pass exactly one of model and options"

        self.equation = equation
        self.options = model.sr_options_ if options is None else options
        self.variable_names = (
            variable_names
            if variable_names is not None
            else (model.feature_names_in_ if model is not None else None)
        )

        from julia import Main, SymbolicRegression

        self.julia_ = Main
        self.backend_ = SymbolicRegression

    @classmethod
    def from_string(
        cls,
        s: str,
        *,
        model: PySRRegressor = None,
        options=None,
        variable_names: Optional[List[str]] = None,
    ):
        self = cls(None, model=model, options=options, variable_names=variable_names)

        for i, variable in enumerate(self.variable_names):
            self.julia_.eval(f"{variable} = Node(feature={i + 1})")

        self.julia_.last_options = self.options
        self.julia_.eval("SymbolicRegression.@extend_operators last_options")

        equation = self.julia_.eval(s)

        if isinstance(equation, Number):
            equation = self.julia_.eval(f"Node(val={equation})")

        self.equation = equation

        return self

    def __repr__(self):
        variable_names = (
            list(self.variable_names) if self.variable_names is not None else None
        )
        return self.backend_.string_tree(
            self.equation, self.options, variable_names=variable_names
        )

    def __call__(self, X):
        return self.equation(X.T, self.options)

    def compute_complexity(self) -> int:
        return int(self.backend_.compute_complexity(self.equation, self.options))
