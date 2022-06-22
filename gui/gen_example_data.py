import pandas as pd
import numpy as np

rand_between = lambda a, b, size: np.random.rand(*size) * (b - a) + a

X = pd.DataFrame(
    {
        "T": rand_between(273, 373, (100,)),  # Kelvin
        "P": rand_between(100, 200, (100,)) * 1e3,  # Pa
        "n": rand_between(0, 10, (100,)),  # mole
    }
)

R = 8.3144598  # J/mol/K
X["y"] = X["n"] * R * X["T"] / X["P"]

X.to_csv("data.csv", index=False)