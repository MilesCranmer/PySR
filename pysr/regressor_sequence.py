import pickle as pkl
import warnings

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
import pandas as pd

from .sr import PySRRegressor
from .utils import (
    ArrayLike,
    PathLike,
)
from .export_latex import (
    sympy2latex,
    sympy2latextable,
    sympy2multilatextable,
    with_preamble,
)


def _check_assertions(
    X,
    recursive_history_length=None,
    weights=None,
    variable_names=None,
    X_units=None,
):
    if recursive_history_length is not None and recursive_history_length <= 0:
        raise ValueError(
            "The `recursive_history_length` parameter must be greater than 0 (otherwise it's not recursion)."
        )
    if len(X.shape) > 2:
        raise ValueError(
            "Recursive symbolic regression only supports up to 2D data; please flatten your data first"
        )
    elif len(X) < 2:
        raise ValueError(
            "Recursive symbolic regression requires at least 2 datapoints; if you tried to pass a 1D array, use array.reshape(-1, 1)"
        )
    if len(X) <= recursive_history_length + 1:
        raise ValueError(
            f"Recursive symbolic regression with a history length of {recursive_history_length} requires at least {recursive_history_length + 2} datapoints."
        )
    if isinstance(weights, np.ndarray) and len(weights) != len(X):
        raise ValueError("The length of `weights` must have shape (n_times,).")
    if isinstance(variable_names, list) and len(variable_names) != X.shape[1]:
        raise ValueError(
            "The length of `variable_names` must be equal to the number of features in `X`."
        )
    if isinstance(X_units, list) and len(X_units) != X.shape[1]:
        raise ValueError(
            "The length of `X_units` must be equal to the number of features in `X`."
        )


class PySRSequenceRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """
    High performance symbolic regression for recurrent sequences.
    Based off of the `PySRRegressor` class, but with a preprocessing step for recurrence relations.

    Parameters
    ----------
    recursive_history_length : int
        The number of previous time points to use as input features.
        For example, if `recursive_history_length=2`, then the input features
        will be `[X[0], X[1]]` and the output will be `X[2]`.
        This continues on for all X: [X[n-1], X[n-2]] to predict X[n].
        Must be greater than 0.
    Other parameters and attributes are inherited from `PySRRegressor`.
    """

    def __init__(
        self,
        recursive_history_length: int = 0,
        **kwargs,
    ):
        self._regressor = PySRRegressor(**kwargs)
        super().__init__()
        self.recursive_history_length = recursive_history_length

    def _construct_variable_names(
        self, n_features: int,
        variable_names: Optional[List[str]]
    ):
        if not isinstance(variable_names, list):
            if n_features == 1:
                return [f"xt_{i}" for i in range(self.recursive_history_length, 0, -1)]
            else:
                return [
                    f"x{i}t_{j}"
                    for j in range(self.recursive_history_length, 0, -1)
                    for i in range(n_features)
                ]
        else:
            return [
                i + "t_" + str(j)
                for j in range(self.recursive_history_length, 0, -1)
                for i in variable_names
            ]

    def fit(
        self,
        X,
        weights=None,
        variable_names: Optional[ArrayLike[str]] = None,
        complexity_of_variables: Optional[
            Union[int, float, List[Union[int, float]]]
        ] = None,
        X_units: Optional[ArrayLike[str]] = None,
    ) -> "PySRSequenceRegressor":
        """
        Search for equations to fit the time series dataset and store them in `self.equations_`.

        Parameters
        ----------
        X : ndarray | pandas.DataFrame
            Sequence of shape (n_times, n_features).
        weights : ndarray | pandas.DataFrame
            Weight array of the same shape as `X`.
            Each element is how to weight the mean-square-error loss
            for that particular element of `X`. Alternatively,
            if a custom `loss` was set, it will can be used
            in arbitrary ways.
        variable_names : list[str]
            A list of names for the variables, rather than "x0t_1", "x1t_2", etc.
            If `X` is a pandas dataframe, the column name will be used
            instead of `variable_names`. Cannot contain spaces or special
            characters. Avoid variable names which are also
            function names in `sympy`, such as "N".
            The number of variable names must be equal to (n_features,).
        complexity_of_variables : int | float | list[int] | list[float]
            The complexity of each variable in `X`. If a single value is
            passed, it will be used for all variables. If a list is passed,
            its length must be the same as recurrence_history_length.
        X_units : list[str]
            A list of units for each variable in `X`. Each unit should be
            a string representing a Julia expression. See DynamicQuantities.jl
            https://symbolicml.org/DynamicQuantities.jl/dev/units/ for more
            information.
            Length should be equal to n_features.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_data(X)
        _check_assertions(
            X,
            self.recursive_history_length,
            weights,
            variable_names,
            X_units,
        )
        self.variable_names = variable_names # for latex_table()
        self.n_features = X.shape[1] # for latex_table()

        current_X = X[self.recursive_history_length :]
        historical_X = np.lib.stride_tricks.sliding_window_view(
            X[:-1].flatten(), self.recursive_history_length * X.shape[1]
        )[:: current_X.shape[1], :]
        y_units = X_units
        if isinstance(weights, np.ndarray):
            weights = weights[self.recursive_history_length :]
        variable_names = self._construct_variable_names(
            current_X.shape[1], variable_names
        )

        self._regressor.fit(
            X=historical_X,
            y=current_X,
            weights=weights,
            variable_names=variable_names,
            X_units=X_units,
            y_units=y_units,
            complexity_of_variables=complexity_of_variables,
        )
        """ self._regressor.__dict__["__sklearn_is_fitted__"] = True
        self._regressor.__dict__["selection_mask_"] = self._regressor.selection_mask_
        self._regressor.__dict__["feature_names_in_"] = (
            self._regressor.feature_names_in_
        )
        self._regressor.__dict__["nout_"] = self._regressor.nout_ """
        return self

    def predict(self, X, index=None, extra_predictions=0):
        """
        Predict y from input X using the equation chosen by `model_selection`.

        You may see what equation is used by printing this object. X should
        have the same columns as the training data.

        Parameters
        ----------
        X : ndarray | pandas.DataFrame
            Data of shape `(n_times, n_features)`.
        index : int | list[int]
            If you want to compute the output of an expression using a
            particular row of `self.equations_`, you may specify the index here.
            For multiple output equations, you must pass a list of indices
            in the same order.
        extra_predictions : int
            If you want to predict more than one step into the future, specify
            how many extra predictions you want. For example, if `extra_predictions=2`,
            the model will predict the next two time points after the last time point
            in `X`.

        Returns
        -------
        x_predicted : ndarray of shape (n_samples, n_features)
            Values predicted by substituting `X` into the fitted sequence symbolic
            regression model.

        Raises
        ------
        ValueError
            Raises if the `best_equation` cannot be evaluated.
        """
        X = self._validate_data(X)
        _check_assertions(X, recursive_history_length=self.recursive_history_length)
        historical_X = np.lib.stride_tricks.sliding_window_view(
            X.flatten(), self.recursive_history_length * np.prod(X.shape[1])
        )[:: X.shape[1], :]
        pred = self._regressor.predict(X=historical_X, index=index)
        if extra_predictions > 0:
            output = pred
            previous_points = historical_X[-1]
            # Without this, the model will re-predict the last data point
            pred_once = self._regressor.predict(X=[previous_points], index=index)
            previous_points = previous_points[X.shape[1] :]
            previous_points = np.append(previous_points, pred_once)
            previous_points = previous_points.flatten()
            for _ in range(extra_predictions):
                pred_once = self._regressor.predict(X=[previous_points], index=index)
                previous_points = previous_points[X.shape[1] :]
                previous_points = np.append(previous_points, pred_once)
                previous_points = previous_points.flatten()
                output = np.append(output, pred_once)
            return output.reshape(-1, X.shape[1])
        return pred

    def from_file(
        self,
        cls,
        equation_file: PathLike,
        *pysr_args,
        binary_operators: Optional[List[str]] = None,
        unary_operators: Optional[List[str]] = None,
        n_features_in: Optional[int] = None,
        feature_names_in: Optional[ArrayLike[str]] = None,
        selection_mask: Optional[NDArray[np.bool_]] = None,
        nout: int = 1,
        **pysr_kwargs
    ):
        return self._regressor.from_file(
            cls,
            equation_file,
            *pysr_args,
            binary_operators,
            unary_operators,
            n_features_in,
            feature_names_in,
            selection_mask,
            nout,
            **pysr_kwargs,
        )
    
    def __repr__(self):
        return self._regressor.__repr__().replace("PySRRegressor", "PySRSequenceRegressor")

    def __getstate__(self):
        """
        Handle pickle serialization for PySRSequenceRegressor.

        The Scikit-learn standard requires estimators to be serializable via
        `pickle.dumps()`. However, some attributes do not support pickling
        and need to be hidden, such as the JAX and Torch representations.
        """
        state = self._regressor.__dict__
        show_pickle_warning = not (
            "show_pickle_warnings_" in state and not state["show_pickle_warnings_"]
        )
        state_keys_containing_lambdas = ["extra_sympy_mappings", "extra_torch_mappings"]
        for state_key in state_keys_containing_lambdas:
            if state[state_key] is not None and show_pickle_warning:
                warnings.warn(
                    f"`{state_key}` cannot be pickled and will be removed from the "
                    "serialized instance. When loading the model, please redefine "
                    f"`{state_key}` at runtime."
                )
        state_keys_to_clear = state_keys_containing_lambdas
        pickled_state = {
            key: (None if key in state_keys_to_clear else value)
            for key, value in state.items()
        }
        if ("equations_" in pickled_state) and (
            pickled_state["equations_"] is not None
        ):
            pickled_state["output_torch_format"] = False
            pickled_state["output_jax_format"] = False
            if self._regressor.nout_ == 1:
                pickled_columns = ~pickled_state["equations_"].columns.isin(
                    ["jax_format", "torch_format"]
                )
                pickled_state["equations_"] = (
                    pickled_state["equations_"].loc[:, pickled_columns].copy()
                )
            else:
                pickled_columns = [
                    ~dataframe.columns.isin(["jax_format", "torch_format"])
                    for dataframe in pickled_state["equations_"]
                ]
                pickled_state["equations_"] = [
                    dataframe.loc[:, signle_pickled_columns]
                    for dataframe, signle_pickled_columns in zip(
                        pickled_state["equations_"], pickled_columns
                    )
                ]
        return pickled_state
    
    @property
    def julia_options_(self):
        return self._regressor.julia_options_

    @property
    def julia_state_(self):
        return self._regressor.julia_state_
    
    def get_best(self, index=None):
        return self._regressor.get_best(index=index)
    
    def refresh(self, checkpoint_file: Optional[PathLike] = None) -> None:
        return self._regressor.refresh(checkpoint_file=checkpoint_file)
    
    def sympy(self, index=None):
        return self._regressor.sympy(index=index)
    
    def latex(self, index=None, precision=3):
        return self._regressor.latex(index=index, precision=precision)
    
    def get_hof(self):
        return self._regressor.get_hof()
    
    def latex_table(
        self,
        indices=None,
        precision=3,
        columns=["equation", "complexity", "loss", "score"],
    ):
        """Create a LaTeX/booktabs table for all, or some, of the equations.

        Parameters
        ----------
        indices : list[int] | list[list[int]]
            If you wish to select a particular subset of equations from
            `self.equations_`, give the row numbers here. By default,
            all equations will be used. If there are multiple output
            features, then pass a list of lists.
        precision : int
            The number of significant figures shown in the LaTeX
            representations.
            Default is `3`.
        columns : list[str]
            Which columns to include in the table.
            Default is `["equation", "complexity", "loss", "score"]`.

        Returns
        -------
        latex_table_str : str
            A string that will render a table in LaTeX of the equations.
        """
        self._regressor.refresh()

        if isinstance(self._regressor.equations_, list):
            if indices is not None:
                assert isinstance(indices, list)
                assert isinstance(indices[0], list)
                assert len(indices) == self._regressor.nout_
            variable_names = self.variable_names
            if variable_names is not None:
                variable_names = [variable_name + "t_0" for variable_name in variable_names]
            else:
                variable_names = [f"x{i}t_0" for i in range(self.n_features)]
            table_string = sympy2multilatextable(
                self._regressor.equations_, indices=indices, precision=precision, columns=columns, output_variable_names=variable_names
            )
        elif isinstance(self.equations_, pd.DataFrame):
            if indices is not None:
                assert isinstance(indices, list)
                assert isinstance(indices[0], int)
            if self.variable_names is not None:
                assert len(self.variable_names) == 1
                variable_name = self.variable_names[0] + "t_0"
            else:
                variable_name = "xt_0"
            table_string = sympy2latextable(
                self._regressor.equations_, indices=indices, precision=precision, columns=columns, output_variable_name=variable_name
            )
        else:
            raise ValueError(
                "Invalid type for equations_ to pass to `latex_table`. "
                "Expected a DataFrame or a list of DataFrames."
            )

        return with_preamble(table_string)
    
    @property
    def equations_(self):
        return self._regressor.equations_