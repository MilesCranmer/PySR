from collections import OrderedDict

import gradio as gr
import numpy as np
from data import TEST_EQUATIONS
from gradio.components.base import Component
from plots import plot_example_data, plot_pareto_curve
from processing import processing


class ExampleData:
    def __init__(self, demo: gr.Blocks) -> None:
        with gr.Column():
            self.example_plot = gr.Plot()
        with gr.Column():
            self.test_equation = gr.Radio(
                TEST_EQUATIONS, value=TEST_EQUATIONS[0], label="Test Equation"
            )
            self.num_points = gr.Slider(
                minimum=10,
                maximum=1000,
                value=200,
                label="Number of Data Points",
                step=1,
            )
            self.noise_level = gr.Slider(
                minimum=0, maximum=1, value=0.05, label="Noise Level"
            )
            self.data_seed = gr.Number(value=0, label="Random Seed")

        # Set up plotting:

        eqn_components = [
            self.test_equation,
            self.num_points,
            self.noise_level,
            self.data_seed,
        ]
        for eqn_component in eqn_components:
            eqn_component.change(
                plot_example_data,
                eqn_components,
                self.example_plot,
                show_progress=False,
            )

        demo.load(plot_example_data, eqn_components, self.example_plot)


class UploadData:
    def __init__(self) -> None:
        self.file_input = gr.File(label="Upload a CSV File")
        self.label = gr.Markdown(
            "The rightmost column of your CSV file will be used as the target variable."
        )


class Data:
    def __init__(self, demo: gr.Blocks) -> None:
        with gr.Tab("Example Data"):
            self.example_data = ExampleData(demo)
        with gr.Tab("Upload Data"):
            self.upload_data = UploadData()


class BasicSettings:
    def __init__(self) -> None:
        self.binary_operators = gr.CheckboxGroup(
            choices=["+", "-", "*", "/", "^", "max", "min", "mod", "cond"],
            label="Binary Operators",
            value=["+", "-", "*", "/"],
        )
        self.unary_operators = gr.CheckboxGroup(
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
        self.niterations = gr.Slider(
            minimum=1,
            maximum=1000,
            value=40,
            label="Number of Iterations",
            step=1,
        )
        self.maxsize = gr.Slider(
            minimum=7,
            maximum=100,
            value=20,
            label="Maximum Complexity",
            step=1,
        )
        self.parsimony = gr.Number(
            value=0.0032,
            label="Parsimony Coefficient",
        )


class AdvancedSettings:
    def __init__(self) -> None:
        self.populations = gr.Slider(
            minimum=2,
            maximum=100,
            value=15,
            label="Number of Populations",
            step=1,
        )
        self.population_size = gr.Slider(
            minimum=2,
            maximum=1000,
            value=33,
            label="Population Size",
            step=1,
        )
        self.ncycles_per_iteration = gr.Number(
            value=550,
            label="Cycles per Iteration",
        )
        self.elementwise_loss = gr.Radio(
            ["L2DistLoss()", "L1DistLoss()", "LogitDistLoss()", "HuberLoss()"],
            value="L2DistLoss()",
            label="Loss Function",
        )
        self.adaptive_parsimony_scaling = gr.Number(
            value=20.0,
            label="Adaptive Parsimony Scaling",
        )
        self.optimizer_algorithm = gr.Radio(
            ["BFGS", "NelderMead"],
            value="BFGS",
            label="Optimizer Algorithm",
        )
        self.optimizer_iterations = gr.Slider(
            minimum=1,
            maximum=100,
            value=8,
            label="Optimizer Iterations",
            step=1,
        )
        self.batching = gr.Checkbox(
            value=False,
            label="Batching",
        )
        self.batch_size = gr.Slider(
            minimum=2,
            maximum=1000,
            value=50,
            label="Batch Size",
            step=1,
        )


class GradioSettings:
    def __init__(self) -> None:
        self.plot_update_delay = gr.Slider(
            minimum=1,
            maximum=100,
            value=3,
            label="Plot Update Delay",
        )
        self.force_run = gr.Checkbox(
            value=False,
            label="Ignore Warnings",
        )


class Settings:
    def __init__(self):
        with gr.Tab("Basic Settings"):
            self.basic_settings = BasicSettings()
        with gr.Tab("Advanced Settings"):
            self.advanced_settings = AdvancedSettings()
        with gr.Tab("Gradio Settings"):
            self.gradio_settings = GradioSettings()


class Results:
    def __init__(self):
        with gr.Tab("Pareto Front"):
            self.pareto = gr.Plot()
        with gr.Tab("Predictions"):
            self.predictions_plot = gr.Plot()

        self.df = gr.Dataframe(
            headers=["complexity", "loss", "equation"],
            datatype=["number", "number", "str"],
            wrap=True,
            column_widths=[75, 75, 200],
            interactive=False,
        )

        self.messages = gr.Textbox(label="Messages", value="", interactive=False)


def flatten_attributes(
    component_group, absolute_name: str, d: OrderedDict
) -> OrderedDict:
    if not hasattr(component_group, "__dict__"):
        return d

    for name, elem in component_group.__dict__.items():
        new_absolute_name = absolute_name + "." + name
        if name.startswith("_"):
            # Private attribute
            continue
        elif elem in d.values():
            # Don't duplicate any tiems
            continue
        elif isinstance(elem, Component):
            # Only add components to dict
            d[new_absolute_name] = elem
        else:
            flatten_attributes(elem, new_absolute_name, d)

    return d


class AppInterface:
    def __init__(self, demo: gr.Blocks) -> None:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    self.data = Data(demo)
                with gr.Row():
                    self.settings = Settings()
            with gr.Column():
                self.results = Results()
                self.run = gr.Button()

        # Update plot when dataframe is updated:
        self.results.df.change(
            plot_pareto_curve,
            inputs=[self.results.df, self.settings.basic_settings.maxsize],
            outputs=[self.results.pareto],
            show_progress=False,
        )

        ignore = ["df", "predictions_plot", "pareto", "messages"]
        self.run.click(
            create_processing_function(self, ignore=ignore),
            inputs=[
                v
                for k, v in flatten_attributes(self, "interface", OrderedDict()).items()
                if last_part(k) not in ignore
            ],
            outputs=[
                self.results.df,
                self.results.predictions_plot,
                self.results.messages,
            ],
            show_progress=True,
        )


def last_part(k: str) -> str:
    return k.split(".")[-1]


def create_processing_function(interface: AppInterface, ignore=[]):
    d = flatten_attributes(interface, "interface", OrderedDict())
    keys = [k for k in map(last_part, d.keys()) if k not in ignore]
    _, idx, counts = np.unique(keys, return_index=True, return_counts=True)
    if np.any(counts > 1):
        raise AssertionError("Bad keys: " + ",".join(np.array(keys)[idx[counts > 1]]))

    def f(*components):
        n = len(components)
        assert n == len(keys)
        for output in processing(**{keys[i]: components[i] for i in range(n)}):
            yield output

    return f


def main():
    with gr.Blocks(theme="default") as demo:
        _ = AppInterface(demo)
        demo.launch(debug=True)


if __name__ == "__main__":
    main()
