import gradio as gr
from data import test_equations
from plots import replot, replot_pareto
from processing import processing


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
                "tan",
                "exp",
                "log",
                "square",
                "cube",
                "sqrt",
                "abs",
                "erf",
                "relu",
                "round",
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
                with gr.Tab("Pareto Front"):
                    blocks["pareto"] = gr.Plot()
                with gr.Tab("Predictions"):
                    blocks["predictions_plot"] = gr.Plot()

                blocks["df"] = gr.Dataframe(
                    headers=["complexity", "loss", "equation"],
                    datatype=["number", "number", "str"],
                    wrap=True,
                    column_widths=[75, 75, 200],
                    interactive=False,
                )
                blocks["run"] = gr.Button()

        blocks["run"].click(
            processing,
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
            eqn_component.change(
                replot, eqn_components, blocks["example_plot"], show_progress=False
            )

        # Update plot when dataframe is updated:
        blocks["df"].change(
            replot_pareto,
            inputs=[blocks["df"], blocks["maxsize"]],
            outputs=[blocks["pareto"]],
            show_progress=False,
        )
        demo.load(replot, eqn_components, blocks["example_plot"])

    demo.launch(debug=True)


if __name__ == "__main__":
    main()
