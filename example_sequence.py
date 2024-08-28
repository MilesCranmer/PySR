import numpy as np

X = [[1, 2], [3, 4]]
for i in range(100):
    X.append([X[-1][0] + X[-2][0], X[-1][1] / X[-2][1]])
X = np.array(X)

from pysr import PySRSequenceRegressor

model = PySRSequenceRegressor(
    recursive_history_length=2,  # How many previous values to use
    # All other parameters are the same as PySRRegressor
    model_selection="best",  # Result is mix of simplicity+accuracy
    niterations=40,
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
)

model.fit(X)  # no y needed

print(model)
