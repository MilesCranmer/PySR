"""Code for exporting discovered expressions to numpy"""
import numpy as np
import pandas as pd
from sympy import lambdify


class CallableEquation:
    """Simple wrapper for numpy lambda functions built with sympy"""

    def __init__(self, sympy_symbols, eqn, selection=None, variable_names=None):
        self._sympy = eqn
        self._sympy_symbols = sympy_symbols
        self._selection = selection
        self._variable_names = variable_names
        self._lambda = lambdify(sympy_symbols, eqn)

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
            X = X[:, self._selection]
        return self._lambda(*X.T) * np.ones(expected_shape)
