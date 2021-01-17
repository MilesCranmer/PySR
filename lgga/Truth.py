import numpy as np


class Truth:
    def __init__(self, transformation, model):
        self.transformation = transformation
        self.weights = list(model.coef_) + [model.intercept_]

    def predict(self, X, y):
        transformed = self.transformation.transform(X)
        res = np.zeros(shape=y.shape)
        for w in range(len(self.weights)):
            if w < X.shape[1]:
                res = res + (X[:, w] * self.weights[w])
            elif w == X.shape[1]:
                res = res + (y * self.weights[w])
            else:
                assert w == X.shape[1] + 1
                res = res + self.weights[w]
        return res

    def transform(self, X):
        return self.transformation.transform(X)

    def __str__(self):
        return f"Auxiliary Truth: {self.transformation} with linear coefficients for X, y, 1 {self.weights}"

    def __repr__(self):
        return str(self)

    def julia_string(self):
        """
        Return an expression that sorta creates a julia instances of Truth with these parameters
        Specifically Truth(type, params, weights)
        Julia indexing starts at 1 not 0 so we need to add 1 to all parameter indices
        """
        index = self.transformation.index
        params = self.transformation.get_params()
        return f"Truth({index}, {[param + 1 for param in params]}, {self.weights})"