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

test_equations = ["sin(2*x)/x + 0.1*x"]


def generate_data(s: str, num_points: int, noise_level: float, data_seed: int):
    rstate = np.random.RandomState(data_seed)
    x = rstate.uniform(-10, 10, num_points)
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
    plot_update_delay,
    parsimony,
    populations,
    population_size,
    ncycles_per_iteration,
    elementwise_loss,
    adaptive_parsimony_scaling,
    optimizer_algorithm,
    optimizer_iterations,
    batching,
    batch_size,
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
                equation_file=equation_file,
                parsimony=parsimony,
                populations=populations,
                population_size=population_size,
                ncycles_per_iteration=ncycles_per_iteration,
                elementwise_loss=elementwise_loss,
                adaptive_parsimony_scaling=adaptive_parsimony_scaling,
                optimizer_algorithm=optimizer_algorithm,
                optimizer_iterations=optimizer_iterations,
                batching=batching,
                batch_size=batch_size,
            ),
        )
        process.start()
        last_yield_time = None
        while process.is_alive():
            if equation_file_bkup.exists():
                try:
                    # First, copy the file to a the copy file
                    equation_file_copy = base / "hall_of_fame_copy.csv"
                    os.system(f"cp {equation_file_bkup} {equation_file_copy}")
                    equations = pd.read_csv(equation_file_copy)
                    # Ensure it is pareto dominated, with more complex expressions
                    # having higher loss. Otherwise remove those rows.
                    # TODO: Not sure why this occurs; could be the result of a late copy?
                    equations.sort_values("Complexity", ascending=True, inplace=True)
                    equations.reset_index(inplace=True)
                    bad_idx = []
                    min_loss = None
                    for i in equations.index:
                        if min_loss is None or equations.loc[i, "Loss"] < min_loss:
                            min_loss = float(equations.loc[i, "Loss"])
                        else:
                            bad_idx.append(i)
                    equations.drop(index=bad_idx, inplace=True)

                    while (
                        last_yield_time is not None
                        and time.time() - last_yield_time < plot_update_delay
                    ):
                        time.sleep(0.1)

                    yield equations[["Complexity", "Loss", "Equation"]]

                    last_yield_time = time.time()
                except pd.errors.EmptyDataError:
                    pass

        process.join()


def greet(
    *,
    X,
    y,
    **pysr_kwargs,
):
    import pysr

    model = pysr.PySRRegressor(
        progress=False,
        timeout_in_seconds=1000,
        **pysr_kwargs,
    )
    model.fit(X, y)

    return 0


def _data_layout():
    with gr.Tab("Example Data"):
        # Plot of the example data:
        with gr.Row():
            with gr.Column():
                example_plot = gr.Plot()
            with gr.Column():
                test_equation = gr.Radio(
                    test_equations, value=test_equations[0], label="Test Equation"
                )
                num_points = gr.Slider(
                    minimum=10,
                    maximum=1000,
                    value=200,
                    label="Number of Data Points",
                    step=1,
                )
                noise_level = gr.Slider(
                    minimum=0, maximum=1, value=0.05, label="Noise Level"
                )
                data_seed = gr.Number(value=0, label="Random Seed")
    with gr.Tab("Upload Data"):
        file_input = gr.File(label="Upload a CSV File")
        gr.Markdown(
            "The rightmost column of your CSV file will be used as the target variable."
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
    with gr.Tab("Basic Settings"):
        binary_operators = gr.CheckboxGroup(
            choices=["+", "-", "*", "/", "^", "max", "min", "mod", "cond"],
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
                "sinh",
                "cosh",
                "tanh",
                "atan",
                "asinh",
                "acosh",
                "erf",
                "relu",
                "round",
                "floor",
                "ceil",
                "sign",
            ],
            label="Unary Operators",
            value=["sin"],
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
            maximum=100,
            value=20,
            label="Maximum Complexity",
            step=1,
        )
        parsimony = gr.Number(
            value=0.0032,
            label="Parsimony Coefficient",
        )
    with gr.Tab("Advanced Settings"):
        populations = gr.Slider(
            minimum=2,
            maximum=100,
            value=15,
            label="Number of Populations",
            step=1,
        )
        population_size = gr.Slider(
            minimum=2,
            maximum=1000,
            value=33,
            label="Population Size",
            step=1,
        )
        ncycles_per_iteration = gr.Number(
            value=550,
            label="Cycles per Iteration",
        )
        elementwise_loss = gr.Radio(
            ["L2DistLoss()", "L1DistLoss()", "LogitDistLoss()", "HuberLoss()"],
            value="L2DistLoss()",
            label="Loss Function",
        )
        adaptive_parsimony_scaling = gr.Number(
            value=20.0,
            label="Adaptive Parsimony Scaling",
        )
        optimizer_algorithm = gr.Radio(
            ["BFGS", "NelderMead"],
            value="BFGS",
            label="Optimizer Algorithm",
        )
        optimizer_iterations = gr.Slider(
            minimum=1,
            maximum=100,
            value=8,
            label="Optimizer Iterations",
            step=1,
        )
        # Bool:
        batching = gr.Checkbox(
            value=False,
            label="Batching",
        )
        batch_size = gr.Slider(
            minimum=2,
            maximum=1000,
            value=50,
            label="Batch Size",
            step=1,
        )

    with gr.Tab("Gradio Settings"):
        plot_update_delay = gr.Slider(
            minimum=1,
            maximum=100,
            value=3,
            label="Plot Update Delay",
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
        plot_update_delay=plot_update_delay,
        parsimony=parsimony,
        populations=populations,
        population_size=population_size,
        ncycles_per_iteration=ncycles_per_iteration,
        elementwise_loss=elementwise_loss,
        adaptive_parsimony_scaling=adaptive_parsimony_scaling,
        optimizer_algorithm=optimizer_algorithm,
        optimizer_iterations=optimizer_iterations,
        batching=batching,
        batch_size=batch_size,
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
                    "plot_update_delay",
                    "parsimony",
                    "populations",
                    "population_size",
                    "ncycles_per_iteration",
                    "elementwise_loss",
                    "adaptive_parsimony_scaling",
                    "optimizer_algorithm",
                    "optimizer_iterations",
                    "batching",
                    "batch_size",
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
        demo.load(replot, eqn_components, blocks["example_plot"])

    demo.launch(debug=True)


def replot_pareto(df, maxsize):
    plt.rcParams["font.family"] = "IBM Plex Mono"
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    if len(df) == 0 or "Equation" not in df.columns:
        return fig

    # Plotting the data
    ax.loglog(
        df["Complexity"],
        df["Loss"],
        marker="o",
        linestyle="-",
        color="#333f48",
        linewidth=1.5,
        markersize=6,
    )

    # Set the axis limits
    ax.set_xlim(0.5, maxsize + 1)
    ytop = 2 ** (np.ceil(np.log2(df["Loss"].max())))
    ybottom = 2 ** (np.floor(np.log2(df["Loss"].min() + 1e-20)))
    ax.set_ylim(ybottom, ytop)

    ax.grid(True, which="both", ls="--", linewidth=0.5, color="gray", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Range-frame the plot
    for direction in ["bottom", "left"]:
        ax.spines[direction].set_position(("outward", 10))

    # Delete far ticks
    ax.tick_params(axis="both", which="major", labelsize=10, direction="out", length=5)
    ax.tick_params(axis="both", which="minor", labelsize=8, direction="out", length=3)

    ax.set_xlabel("Complexity")
    ax.set_ylabel("Loss")
    fig.tight_layout(pad=2)

    return fig


def replot(test_equation, num_points, noise_level, data_seed):
    X, y = generate_data(test_equation, num_points, noise_level, data_seed)
    x = X["x"]

    plt.rcParams["font.family"] = "IBM Plex Mono"
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    ax.scatter(x, y, alpha=0.7, edgecolors="w", s=50)

    ax.grid(True, which="both", ls="--", linewidth=0.5, color="gray", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Range-frame the plot
    for direction in ["bottom", "left"]:
        ax.spines[direction].set_position(("outward", 10))

    # Delete far ticks
    ax.tick_params(axis="both", which="major", labelsize=10, direction="out", length=5)
    ax.tick_params(axis="both", which="minor", labelsize=8, direction="out", length=3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout(pad=2)

    return fig


if __name__ == "__main__":
    main()
