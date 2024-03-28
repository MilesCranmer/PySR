import gradio as gr
import numpy as np
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

test_equations = [
    "sin(x) + cos(2*x) + tan(x/3)",
]


def generate_data(s: str, num_points: int, noise_level: float):
    x = np.linspace(0, 10, num_points)
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


def _data_layout():
    with gr.Tab("Example Data"):
        # Plot of the example data:
        example_plot = gr.ScatterPlot(
            x="x",
            y="y",
            tooltip=["x", "y"],
            x_lim=[0, 10],
            y_lim=[-5, 5],
            width=350,
            height=300,
        )
        test_equation = gr.Radio(
            test_equations, value=test_equations[0], label="Test Equation"
        )
        num_points = gr.Slider(
            minimum=10,
            maximum=1000,
            value=100,
            label="Number of Data Points",
            step=1,
        )
        noise_level = gr.Slider(minimum=0, maximum=1, value=0.1, label="Noise Level")
    with gr.Tab("Upload Data"):
        file_input = gr.File(label="Upload a CSV File")
        gr.Markdown(
            "Upload a CSV file with the data to fit. The last column will be used as the target variable."
        )

    return dict(
        file_input=file_input,
        test_equation=test_equation,
        num_points=num_points,
        noise_level=noise_level,
        example_plot=example_plot,
    )


def _settings_layout():
    binary_operators = gr.CheckboxGroup(
        choices=["+", "-", "*", "/", "^"],
        label="Binary Operators",
        value=["+", "-", "*", "/"],
    )
    unary_operators = gr.CheckboxGroup(
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
    )
    niterations = gr.Slider(
        minimum=1,
        maximum=1000,
        value=40,
        label="Number of Iterations",
        step=1,
    )
    maxsize = gr.Slider(
        minimum=7,
        maximum=35,
        value=20,
        label="Maximum Complexity",
        step=1,
    )
    force_run = gr.Checkbox(
        value=False,
        label="Ignore Warnings",
    )
    return dict(
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        niterations=niterations,
        maxsize=maxsize,
        force_run=force_run,
    )


def main():
    blocks = {}
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    blocks = {**blocks, **_data_layout()}
                with gr.Row():
                    blocks = {**blocks, **_settings_layout()}

            with gr.Column():
                blocks["df"] = gr.Dataframe(
                    headers=["Equation", "Loss", "Complexity"],
                    datatype=["str", "number", "number"],
                )
                blocks["run"] = gr.Button()
                blocks["error_log"] = gr.Textbox(label="Error Log")

        blocks["run"].click(
            greet,
            inputs=[
                blocks[k]
                for k in [
                    "file_input",
                    "test_equation",
                    "num_points",
                    "noise_level",
                    "niterations",
                    "maxsize",
                    "binary_operators",
                    "unary_operators",
                    "force_run",
                ]
            ],
            outputs=[blocks["df"], blocks["error_log"]],
        )

        # Any update to the equation choice will trigger a replot:
        eqn_components = [
            blocks["test_equation"],
            blocks["num_points"],
            blocks["noise_level"],
        ]
        for eqn_component in eqn_components:
            eqn_component.change(replot, eqn_components, blocks["example_plot"])

    demo.launch()


def replot(test_equation, num_points, noise_level):
    X, y = generate_data(test_equation, num_points, noise_level)
    df = pd.DataFrame({"x": X["x"], "y": y})
    return df


if __name__ == "__main__":
    main()
