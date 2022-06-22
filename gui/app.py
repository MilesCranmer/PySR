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

def greet(
    file_obj: tempfile._TemporaryFileWrapper,
    col_to_fit: str,
    niterations: int,
    binary_operators: list,
    unary_operators: list,
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

    os.system("bash install_pysr.sh")
    os.system(f"python run_pysr_and_save.py --niterations {niterations} --binary_operators '{binary_operators}' --unary_operators '{unary_operators}' --col_to_fit {col_to_fit} --filename {file_obj.name}")
    df = pd.read_csv("pysr_output.csv")
    error_log = open("error.log", "r").read()
    return df, error_log


def main():
    demo = gr.Interface(
        fn=greet,
        description="PySR Demo",
        inputs=[
            gr.inputs.File(label="Upload a CSV File"),
            gr.inputs.Textbox(label="Column to Predict", placeholder="y"),
            gr.inputs.Slider(
                minimum=1,
                maximum=1000,
                default=40,
                label="Number of iterations",
            ),
            gr.inputs.CheckboxGroup(
                choices=["+", "-", "*", "/", "^"],
                label="Binary Operators",
                default=["+", "-", "*", "/"],
            ),
            gr.inputs.CheckboxGroup(
                choices=["sin", "cos", "exp", "log"],
                label="Unary Operators",
                default=[],
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
