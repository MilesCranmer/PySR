import gradio as gr
import os
import tempfile
import pandas as pd

empty_df = pd.DataFrame(
    {
        "equation": [],
        "loss": [],
        "complexity": [],
    }
)

os.system("bash install_pysr.sh")


def greet(
    file_obj: tempfile._TemporaryFileWrapper,
    col_to_fit: str,
    niterations: int,
    maxsize: int,
    binary_operators: list,
    unary_operators: list,
    force_run: bool,
):
    if col_to_fit == "":
        return (
            empty_df,
            "Please enter a column to predict!",
        )
    if len(binary_operators) == 0 and len(unary_operators) == 0:
        return (
            empty_df,
            "Please select at least one operator!",
        )
    if file_obj is None:
        return (
            empty_df,
            "Please upload a CSV file!",
        )
    # Look at some statistics of the file:
    df = pd.read_csv(file_obj.name)
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
    if col_to_fit not in df.columns:
        return (
            empty_df,
            f"The column to predict, {col_to_fit}, is not in the file!"
            f"I found {df.columns}.",
        )
    if len(df) > 1000 and not force_run:
        return (
            empty_df,
            "You have uploaded a file with more than 2000 rows. "
            "This will take very long to run. "
            "Please upload a subsample of the data, "
            "or check the box 'Ignore Warnings'.",
        )

    binary_operators = str(binary_operators).replace("'", '"')
    unary_operators = str(unary_operators).replace("'", '"')
    os.system(
        f"python run_pysr_and_save.py "
        f"--niterations {niterations} "
        f"--maxsize {maxsize} "
        f"--binary_operators '{binary_operators}' "
        f"--unary_operators '{unary_operators}' "
        f"--col_to_fit {col_to_fit} "
        f"--filename {file_obj.name}"
    )
    df = pd.read_csv("pysr_output.csv")
    error_log = open("error.log", "r").read()
    return df, error_log


def main():
    demo = gr.Interface(
        fn=greet,
        description="Symbolic Regression with PySR. Watch search progress by clicking 'See logs'!",
        inputs=[
            gr.inputs.File(label="Upload a CSV File"),
            gr.inputs.Textbox(label="Column to Predict", placeholder="y"),
            gr.inputs.Slider(
                minimum=1,
                maximum=1000,
                default=40,
                label="Number of Iterations",
                step=1,
            ),
            gr.inputs.Slider(
                minimum=7,
                maximum=35,
                default=20,
                label="Maximum Complexity",
                step=1,
            ),
            gr.inputs.CheckboxGroup(
                choices=["+", "-", "*", "/", "^"],
                label="Binary Operators",
                default=["+", "-", "*", "/"],
            ),
            gr.inputs.CheckboxGroup(
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
                default=[],
            ),
            gr.inputs.Checkbox(
                default=False,
                label="Ignore Warnings",
            ),
        ],
        outputs=[
            "dataframe",
            gr.outputs.Textbox(label="Error Log"),
        ],
    )
    # Add file to the demo:

    demo.launch()


if __name__ == "__main__":
    main()
