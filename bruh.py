import numpy as np
from pysr import PySRSequenceRegressor

X = [
        [1, 2, 3],
        [8, 7, 6],
]
for i in range(2, 10):
    X.append([
        X[i-1][2] * X[i-2][1],
        X[i-2][1] - X[i-1][0],
        X[i-1][2] / X[i-1][0],
    ])
X = np.asarray(X)
print(X)
model = PySRSequenceRegressor(
    recursive_history_length=2,
    early_stop_condition="stop_if(loss, complexity) = loss < 1e-4 && complexity == 1",
)
model.fit(X,variable_names=["x", "y", "z"])
print(model.equations_)