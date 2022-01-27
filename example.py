import numpy as np

X = 2 * np.random.randn(100, 5)
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5

from pysr import PySRRegressor

model = PySRRegressor(
    niterations=5,
    populations=8,
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
    ],
    model_selection="best",
)

model.fit(X, y)

print(model)
