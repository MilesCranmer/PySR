import os
import pandas as pd
import traceback as tb
import numpy as np
from argparse import ArgumentParser

# Args:
# niterations
# binary_operators
# unary_operators
# col_to_fit

empty_df = pd.DataFrame(
    {
        "equation": [],
        "loss": [],
        "complexity": [],
    }
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("niterations", type=int)
    parser.add_argument("binary_operators", type=str)
    parser.add_argument("unary_operators", type=str)
    parser.add_argument("col_to_fit", type=str)
    parser.add_argument("filename", type=str)
    args = parser.parse_args()
    niterations = args.niterations
    binary_operators = eval(args.binary_operators)
    unary_operators = eval(args.unary_operators)
    col_to_fit = args.col_to_fit
    filename = args.filename

    os.environ["PATH"] += ":/home/user/.local/bin/"

    try:
        import pysr
        from julia.api import JuliaInfo
        info = JuliaInfo.load(julia="/home/user/.local/bin/julia")
        from julia import Main as _Main
        pysr.sr.Main = _Main

        from pysr import PySRRegressor

        df = pd.read_csv(filename)
        y = np.array(df[col_to_fit])
        X = df.drop([col_to_fit], axis=1)

        model = PySRRegressor(
            update=False,
            niterations=niterations,
            binary_operators=binary_operators,
            unary_operators=unary_operators,
        )
        model.fit(X, y)

        df = model.equations_[["equation", "loss", "complexity"]]
        # Convert all columns to string type:
        df = df.astype(str)
        df.to_csv("pysr_output.csv", index=False)
    except Exception as e:
        error_message = tb.format_exc()
        # Dump to file:
        empty_df.to_csv("pysr_output.csv", index=False)
        with open("error.log", "w") as f:
            f.write(error_message)
    