import gradio as gr
import numpy as np
import pandas as pd
import multiprocessing as mp
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


def _greet_dispatch(
    file_input,
    force_run,
    test_equation,
    num_points,
    noise_level,
    niterations,
    maxsize,
    binary_operators,
    unary_operators,
    seed,
):
    """Load data, then spawn a process to run the greet function."""
    if file_input is not None:
        # Look at some statistics of the file:
        df = pd.read_csv(file_input)
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
        # X, y = generate_data(block["test_equation"], block["num_points"], block["noise_level"])
        X, y = generate_data(test_equation, num_points, noise_level)

    queue = mp.Queue()
    process = mp.Process(
        target=greet,
        kwargs=dict(
            X=X,
            y=y,
            queue=queue,
            niterations=niterations,
            maxsize=maxsize,
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            seed=seed,
        ),
    )
    process.start()
    output = queue.get()
    process.join()
    return output


def greet(
    *,
    queue: mp.Queue,
    X,
    y,
    niterations: int,
    maxsize: int,
    binary_operators: list,
    unary_operators: list,
    seed: int,
):
    import pysr

    model = pysr.PySRRegressor(
        progress=False,
        maxsize=maxsize,
        niterations=niterations,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        timeout_in_seconds=1000,
        multithreading=False,
        procs=0,
        deterministic=True,
        random_state=seed,
    )
    model.fit(X, y)

    df = model.equations_[["complexity", "loss", "equation"]]
    # Convert all columns to string type:
    queue.put(df)

    return 0


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
    seed = gr.Number(
        value=0,
        label="Random Seed",
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
        seed=seed,
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
                    headers=["complexity", "loss", "equation"],
                    datatype=["number", "number", "str"],
                )
                blocks["run"] = gr.Button()

        blocks["run"].click(
            _greet_dispatch,
            inputs=[
                blocks[k]
                for k in [
                    "file_input",
                    "force_run",
                    "test_equation",
                    "num_points",
                    "noise_level",
                    "niterations",
                    "maxsize",
                    "binary_operators",
                    "unary_operators",
                    "seed",
                ]
            ],
            outputs=[blocks["df"]],
        )

        # Any update to the equation choice will trigger a replot:
        eqn_components = [
            blocks["test_equation"],
            blocks["num_points"],
            blocks["noise_level"],
        ]
        for eqn_component in eqn_components:
            eqn_component.change(replot, eqn_components, blocks["example_plot"])

    demo.launch(debug=True)


def replot(test_equation, num_points, noise_level):
    X, y = generate_data(test_equation, num_points, noise_level)
    df = pd.DataFrame({"x": X["x"], "y": y})
    return df


if __name__ == "__main__":
    main()
