import gradio as gr
import numpy as np
import os
import pandas as pd
import time
import multiprocessing as mp
from matplotlib import pyplot as plt
plt.ioff()
import tempfile
from typing import Optional, Union
from pathlib import Path

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


def generate_data(s: str, num_points: int, noise_level: float, data_seed: int):
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
    rstate = np.random.RandomState(data_seed)
    noise = rstate.normal(0, noise_level, y.shape)
    y_noisy = y + noise
    return pd.DataFrame({"x": x}), y_noisy


def _greet_dispatch(
    file_input,
    force_run,
    test_equation,
    num_points,
    noise_level,
    data_seed,
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
        X, y = generate_data(test_equation, num_points, noise_level, data_seed)

    with tempfile.TemporaryDirectory() as tmpdirname:
        base = Path(tmpdirname)
        equation_file = base / "hall_of_fame.csv"
        equation_file_bkup = base / "hall_of_fame.csv.bkup"
        process = mp.Process(
            target=greet,
            kwargs=dict(
                X=X,
                y=y,
                niterations=niterations,
                maxsize=maxsize,
                binary_operators=binary_operators,
                unary_operators=unary_operators,
                seed=seed,
                equation_file=equation_file,
            ),
        )
        process.start()
        while process.is_alive():
            if equation_file_bkup.exists():
                try:
                    # First, copy the file to a the copy file
                    equation_file_copy = base / "hall_of_fame_copy.csv"
                    os.system(f"cp {equation_file_bkup} {equation_file_copy}")
                    df = pd.read_csv(equation_file_copy)
                    # Ensure it is pareto dominated, with more complex expressions
                    # having higher loss. Otherwise remove those rows.
                    # TODO: Not sure why this occurs; could be the result of a late copy?
                    df.sort_values("Complexity", ascending=True, inplace=True)
                    df.reset_index(inplace=True)
                    bad_idx = []
                    min_loss = None
                    for i in df.index:
                        if min_loss is None or df.loc[i, "Loss"] < min_loss:
                            min_loss = float(df.loc[i, "Loss"])
                        else:
                            bad_idx.append(i)
                    df.drop(index=bad_idx, inplace=True)
                    yield df[["Complexity", "Loss", "Equation"]]
                except pd.errors.EmptyDataError:
                    pass
            time.sleep(1)

        process.join()


def greet(
    *,
    X,
    y,
    niterations: int,
    maxsize: int,
    binary_operators: list,
    unary_operators: list,
    seed: int,
    equation_file: Union[str, Path],
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
        equation_file=equation_file,
    )
    model.fit(X, y)

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
        data_seed = gr.Number(value=0, label="Random Seed")
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
        data_seed=data_seed,
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
                blocks["pareto"] = gr.Plot()
                blocks["df"] = gr.Dataframe(
                    headers=["complexity", "loss", "equation"],
                    datatype=["number", "number", "str"],
                    wrap=True,
                    column_widths=[100, 100, 300],
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
                    "data_seed",
                    "niterations",
                    "maxsize",
                    "binary_operators",
                    "unary_operators",
                    "seed",
                ]
            ],
            outputs=blocks["df"],
        )

        # Any update to the equation choice will trigger a replot:
        eqn_components = [
            blocks["test_equation"],
            blocks["num_points"],
            blocks["noise_level"],
            blocks["data_seed"],
        ]
        for eqn_component in eqn_components:
            eqn_component.change(replot, eqn_components, blocks["example_plot"])

        # Update plot when dataframe is updated:
        blocks["df"].change(
            replot_pareto,
            inputs=[blocks["df"], blocks["maxsize"]],
            outputs=[blocks["pareto"]],
        )

    demo.launch(debug=True)


def replot(test_equation, num_points, noise_level, data_seed):
    X, y = generate_data(test_equation, num_points, noise_level, data_seed)
    df = pd.DataFrame({"x": X["x"], "y": y})
    return df

def replot_pareto(df, maxsize):
    # Matplotlib log-log plot of loss vs complexity:
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.set_xlabel('Complexity', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    if len(df) == 0 or 'Equation' not in df.columns:
        return fig

    ax.loglog(df['Complexity'], df['Loss'], marker='o', linestyle='-', color='b')
    ax.set_xlim(1, maxsize + 1)
    # Set ylim to next power of 2:
    ytop = 2 ** (np.ceil(np.log2(df['Loss'].max())))
    ybottom = 2 ** (np.floor(np.log2(df['Loss'].min() + 1e-20)))
    ax.set_ylim(ybottom, ytop)
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    fig.tight_layout()
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    return fig

def replot_pareto(df, maxsize):
    plt.rcParams['font.family'] = 'IBM Plex Mono'
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    if len(df) == 0 or 'Equation' not in df.columns:
        return fig

    # Plotting the data
    ax.loglog(df['Complexity'], df['Loss'], marker='o', linestyle='-', color='#333f48', linewidth=1.5, markersize=6)

    # Set the axis limits
    ax.set_xlim(0.5, maxsize + 1)
    ytop = 2 ** (np.ceil(np.log2(df['Loss'].max())))
    ybottom = 2 ** (np.floor(np.log2(df['Loss'].min() + 1e-20)))
    ax.set_ylim(ybottom, ytop)

    ax.grid(True, which="both", ls="--", linewidth=0.5, color='gray', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Range-frame the plot
    for direction in ['bottom', 'left']:
        ax.spines[direction].set_position(('outward', 10))

    # Delete far ticks
    ax.tick_params(axis='both', which='major', labelsize=10, direction='out', length=5)
    ax.tick_params(axis='both', which='minor', labelsize=8, direction='out', length=3)

    ax.set_xlabel('Complexity')
    ax.set_ylabel('Loss')
    fig.tight_layout(pad=2)

    return fig

if __name__ == "__main__":
    main()
