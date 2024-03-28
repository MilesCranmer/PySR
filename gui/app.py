import gradio as gr
import numpy as np
import os
import pandas as pd
import pysr
import tempfile
from typing import Optional

empty_df = pd.DataFrame(
    {
        "equation": [],
        "loss": [],
        "complexity": [],
    }
)

test_equations = {
    "Complex Polynomial": "3*x^3 + 2*x^2 - x + sin(x)",
    "Exponential and Logarithmic": "exp(-x) + log(x+1)",
    "Trigonometric Polynomial": "sin(x) + cos(2*x) + tan(x/3)",
    "Mixed Functions": "sqrt(x)*exp(-x) + cos(pi*x)",
    "Rational Function": "(x^2 + 1) / (x - 2)",
}


def generate_data(equation: str, num_points: int, noise_level: float):
    x = np.linspace(-10, 10, num_points)
    s = test_equations[equation]
    for (k, v) in {
        "sin": "np.sin",
        "cos": "np.cos",
        "exp": "np.exp",
        "log": "np.log",
        "tan": "np.tan",
        "^": "**",
    }.items():
        s = s.replace(k, v)
    y = eval(s)
    noise = np.random.normal(0, noise_level, y.shape)
    y_noisy = y + noise
    return pd.DataFrame({"x": x}), y_noisy


def greet(
    file_obj: Optional[tempfile._TemporaryFileWrapper],
    test_equation: str,
    num_points: int,
    noise_level: float,
    niterations: int,
    maxsize: int,
    binary_operators: list,
    unary_operators: list,
    force_run: bool,
):
    if file_obj is not None:
        if len(binary_operators) == 0 and len(unary_operators) == 0:
            return (
                empty_df,
                "Please select at least one operator!",
            )
        # Look at some statistics of the file:
        df = pd.read_csv(file_obj)
        if len(df) == 0:
            return (
                empty_df,
                "The file is empty!",
            )
        if len(df.columns) == 1:
            return (
                empty_df,
                "The file has only one column!",
            )
        if len(df) > 10_000 and not force_run:
            return (
                empty_df,
                "You have uploaded a file with more than 10,000 rows. "
                "This will take very long to run. "
                "Please upload a subsample of the data, "
                "or check the box 'Ignore Warnings'.",
            )

        col_to_fit = df.columns[-1]
        y = np.array(df[col_to_fit])
        X = df.drop([col_to_fit], axis=1)
    else:
        X, y = generate_data(test_equation, num_points, noise_level)

    model = pysr.PySRRegressor(
        bumper=True,
        maxsize=maxsize,
        niterations=niterations,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        timeout_in_seconds=1000,
    )
    model.fit(X, y)

    df = model.equations_[["equation", "loss", "complexity"]]
    # Convert all columns to string type:
    df = df.astype(str)
    msg = (
        "Success!\n"
        f"You may run the model locally (faster) with "
        f"the following parameters:"
        + f"""
model = PySRRegressor(
    niterations={niterations},
    binary_operators={str(binary_operators)},
    unary_operators={str(unary_operators)},
    maxsize={maxsize},
)
model.fit(X, y)"""
    )

    df.to_csv("pysr_output.csv", index=False)
    return df, msg


def main():
    demo = gr.Interface(
        fn=greet,
        description="Symbolic Regression with PySR. Watch search progress by following the logs.",
        inputs=[
            gr.File(label="Upload a CSV File"),
            gr.Radio(list(test_equations.keys()), label="Test Equation"),
            gr.Slider(
                minimum=10,
                maximum=1000,
                value=100,
                label="Number of Data Points",
                step=1,
            ),
            gr.Slider(minimum=0, maximum=1, value=0.1, label="Noise Level"),
            gr.Slider(
                minimum=1,
                maximum=1000,
                value=40,
                label="Number of Iterations",
                step=1,
            ),
            gr.Slider(
                minimum=7,
                maximum=35,
                value=20,
                label="Maximum Complexity",
                step=1,
            ),
            gr.CheckboxGroup(
                choices=["+", "-", "*", "/", "^"],
                label="Binary Operators",
                value=["+", "-", "*", "/"],
            ),
            gr.CheckboxGroup(
                choices=[
                    "sin",
                    "cos",
                    "exp",
                    "log",
                    "square",
                    "cube",
                    "sqrt",
                    "abs",
                    "tan",
                ],
                label="Unary Operators",
                value=[],
            ),
            gr.Checkbox(
                value=False,
                label="Ignore Warnings",
            ),
        ],
        outputs=[
            "dataframe",
            gr.Textbox(label="Error Log"),
        ],
    )
    # Add file to the demo:

    demo.launch()


if __name__ == "__main__":
    main()
