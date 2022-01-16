from pysr import pysr, best_row
from sklearn.base import BaseEstimator


class PySRRegressor(BaseEstimator):
    def __init__(self, model_selection="accuracy", **params):
        """Initialize settings for pysr.pysr call.

        :param model_selection: How to select a model. Can be 'accuracy' or 'best'. 'best' will optimize a combination of complexity and accuracy.
        :type model_selection: str
        """
        super().__init__()
        self.model_selection = model_selection
        self.params = params

        # Stored equations:
        self.equations = None

    def __repr__(self):
        return f"PySRRegressor(equations={self.get_best()['sympy_format']})"

    def set_params(self, **params):
        """Set parameters for pysr.pysr call or model_selection strategy."""
        for key, value in params.items():
            if key == "model_selection":
                self.model_selection = value
            self.params[key] = value

        return self

    def get_params(self, deep=True):
        del deep
        return {**self.params, "model_selection": self.model_selection}

    def get_best(self):
        if self.equations is None:
            return 0.0
        if self.model_selection == "accuracy":
            return self.equations.iloc[-1]
        elif self.model_selection == "best":
            return best_row(self.equations)
        else:
            raise NotImplementedError

    def fit(self, X, y):
        self.equations = pysr(
            X=X,
            y=y,
            **self.params,
        )
        return self

    def predict(self, X):
        equation_row = self.get_best()
        np_format = equation_row["lambda_format"]

        return np_format(X)
