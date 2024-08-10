import os
import pickle as pkl
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin

from .export_latex import (
    sympy2latex,
    sympy2latextable,
    sympy2multilatextable,
    with_preamble,
)
from .sr import PySRRegressor
from .utils import (
    ArrayLike,
    PathLike,
    _csv_filename_to_pkl_filename,
    _preprocess_julia_floats,
    _safe_check_feature_names_in,
    _subscriptify,
    _suggest_keywords,
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
        self, n_features: int, variable_names: Optional[List[str]]
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
        Search for equations to fit the sequence and store them in `self.equations_`.

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
        self.variable_names = variable_names  # for latex_table()
        self.n_features = X.shape[1]  # for latex_table()

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

    @classmethod
    def from_file(
        cls,
        equation_file: PathLike,
        *pysr_args,
        recursive_history_length: int,
        binary_operators: Optional[List[str]] = None,
        unary_operators: Optional[List[str]] = None,
        n_features_in: Optional[int] = None,
        feature_names_in: Optional[ArrayLike[str]] = None,
        selection_mask: Optional[NDArray[np.bool_]] = None,
        nout: int = 1,
        **pysr_kwargs,
    ):
        pkl_filename = _csv_filename_to_pkl_filename(equation_file)

        assert recursive_history_length is not None and recursive_history_length > 0

        # Try to load model from <equation_file>.pkl
        print(f"Checking if {pkl_filename} exists...")
        if os.path.exists(pkl_filename):
            print(f"Loading model from {pkl_filename}")
            assert binary_operators is None
            assert unary_operators is None
            assert n_features_in is None
            model = cls(*pysr_args, **pysr_kwargs)
            with open(pkl_filename, "rb") as f:
                model._regressor = pkl.load(f)
            # Change equation_file_ to be in the same dir as the pickle file
            base_dir = os.path.dirname(pkl_filename)
            base_equation_file = os.path.basename(model._regressor.equation_file_)
            model._regressor.equation_file_ = os.path.join(base_dir, base_equation_file)

            # Update any parameters if necessary, such as
            # extra_sympy_mappings:
            model._regressor.set_params(**pysr_kwargs)
            if "equations_" not in model.__dict__ or model.equations_ is None:
                model._regressor.refresh()

            model.recursive_history_length = recursive_history_length
            return model

        # Else, we re-create it.
        print(
            f"{pkl_filename} does not exist, "
            "so we must create the model from scratch."
        )
        assert binary_operators is not None or unary_operators is not None
        assert n_features_in is not None
        assert recursive_history_length is not None

        # TODO: copy .bkup file if exists.
        model = cls(
            equation_file=str(equation_file),
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            **pysr_kwargs,
        )

        model._regressor.nout_ = nout
        model._regressor.n_features_in_ = n_features_in

        if feature_names_in is None:
            model._regressor.feature_names_in_ = np.array(
                [f"x{i}" for i in range(n_features_in)]
            )
            model._regressor.display_feature_names_in_ = np.array(
                [f"x{_subscriptify(i)}" for i in range(n_features_in)]
            )
        else:
            assert len(feature_names_in) == n_features_in
            model._regressor.feature_names_in_ = feature_names_in
            model._regressor.display_feature_names_in_ = feature_names_in

        if selection_mask is None:
            model._regressor.selection_mask_ = np.ones(n_features_in, dtype=np.bool_)
        else:
            model._regressor.selection_mask_ = selection_mask

        model._regressor.refresh(checkpoint_file=equation_file)
        model.recursive_history_length = recursive_history_length

        return model

    def __repr__(self):
        return self._regressor.__repr__().replace(
            "PySRRegressor", "PySRSequenceRegressor"
        )

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
                variable_names = [
                    variable_name + "t_0" for variable_name in variable_names
                ]
            else:
                variable_names = [f"x{i}t_0" for i in range(self.n_features)]
            table_string = sympy2multilatextable(
                self._regressor.equations_,
                indices=indices,
                precision=precision,
                columns=columns,
                output_variable_names=variable_names,
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
                self._regressor.equations_,
                indices=indices,
                precision=precision,
                columns=columns,
                output_variable_name=variable_name,
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
