import numpy as np
import pandas as pd

test_equations = ["sin(2*x)/x + 0.1*x"]


def generate_data(s: str, num_points: int, noise_level: float, data_seed: int):
    rstate = np.random.RandomState(data_seed)
    x = rstate.uniform(-10, 10, num_points)
    for k, v in {
        "sin": "np.sin",
        "cos": "np.cos",
        "exp": "np.exp",
        "log": "np.log",
        "tan": "np.tan",
        "^": "**",
    }.items():
        s = s.replace(k, v)
    y = eval(s)
    noise = rstate.normal(0, noise_level, y.shape)
    y_noisy = y + noise
    return pd.DataFrame({"x": x}), y_noisy


def read_csv(file_input: str, force_run: bool):
    # Look at some statistics of the file:
    df = pd.read_csv(file_input)
    if len(df) == 0:
        raise ValueError("The file is empty!")
    if len(df.columns) == 1:
        raise ValueError("The file has only one column!")
    if len(df) > 10_000 and not force_run:
        raise ValueError(
            "You have uploaded a file with more than 10,000 rows. "
            "This will take very long to run. "
            "Please upload a subsample of the data, "
            "or check the box 'Ignore Warnings'.",
        )

    col_to_fit = df.columns[-1]
    y = np.array(df[col_to_fit])
    X = df.drop([col_to_fit], axis=1)

    return X, y
