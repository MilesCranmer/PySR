import numpy as np
from pysr import PySRRegressor

# Dataset
X = 3 * np.random.randn(100, 5)
y = 3 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 2

# Learn equations
model = PySRRegressor(
    niterations=6,
    binary_operators=["plus", "mult"],
    unary_operators=[
        "cos",
        "exp",
        "sin",  # Pre-defined library of operators (see https://pysr.readthedocs.io/en/latest/docs/operators/)
        "inv(x) = 2/x",
    ],
    loss="loss(x, y) = abs(x - y)",  # Custom loss function
)  # Define your own operator! (Julia syntax)

model.fit(X, y)

print(model)