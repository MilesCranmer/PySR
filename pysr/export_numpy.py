"""Code for exporting discovered expressions to numpy"""
import numpy as np
import pandas as pd
from sympy import lambdify
import warnings


class CallableEquation:
    """Simple wrapper for numpy lambda functions built with sympy"""

    def __init__(self, sympy_symbols, eqn, selection=None, variable_names=None):
        self._sympy = eqn
        self._sympy_symbols = sympy_symbols
        self._selection = selection
        self._variable_names = variable_names

    def __repr__(self):
        return f"PySRFunction(X=>{self._sympy})"

    def __call__(self, X):
        expected_shape = (X.shape[0],)
        if isinstance(X, pd.DataFrame):
            # Lambda function takes as argument:
            return self._lambda(
                **{k: X[k].values for k in self._variable_names}
            ) * np.ones(expected_shape)
        if self._selection is not None:
            if X.shape[1] != len(self._selection):
                warnings.warn(
                    "`X` should be of shape (n_samples, len(self._selection)). "
                    "Automatically filtering `X` to selection. "
                    "Note: Filtered `X` column order may not match column order in fit "
                    "this may lead to incorrect predictions and other errors."
                )
                X = X[:, self._selection]
        return self._lambda(*X.T) * np.ones(expected_shape)

    @property
    def _lambda(self):
        return lambdify(self._sympy_symbols, self._sympy)
