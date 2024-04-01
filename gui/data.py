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
