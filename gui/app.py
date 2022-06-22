import io
import gradio as gr
import os
import tempfile
import numpy as np
import pandas as pd


def greet(
    file_obj: tempfile._TemporaryFileWrapper,
    col_to_fit: str,
    niterations: int,
    binary_operators: list,
    unary_operators: list,
):
    empty_df = pd.DataFrame(
        {
            "equation": [],
            "loss": [],
            "complexity": [],
        }
    )
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
    niterations = int(niterations)
    # Need to install PySR in separate python instance:
    os.system(
        """if [ ! -d "$HOME/.julia/environments/pysr-0.9.1" ]
    then
        python -c 'import pysr; pysr.install()'
    fi"""
    )
    from pysr import PySRRegressor

    df = pd.read_csv(file_obj.name)
    y = np.array(df[col_to_fit])
    X = df.drop([col_to_fit], axis=1)

    model = PySRRegressor(
        update=False,
        temp_equation_file=True,
        niterations=niterations,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
    )
    model.fit(X, y)

    df = model.equations_[["equation", "loss", "complexity"]]
    # Convert all columns to string type:
    df = df.astype(str)
    return df, "Successful."


def main():
    demo = gr.Interface(
        fn=greet,
        description="A demo of PySR",
        inputs=[
            gr.File(label="Upload a CSV File"),
            gr.Textbox(label="Column to Predict", placeholder="y"),
            gr.Slider(
                minimum=1,
                maximum=1000,
                value=40,
                label="Number of iterations",
            ),
            gr.CheckboxGroup(
                choices=["+", "-", "*", "/", "^"],
                label="Binary Operators",
                value=["+", "-", "*", "/"],
            ),
            gr.CheckboxGroup(
                choices=["sin", "cos", "exp", "log"],
                label="Unary Operators",
                value=[],
            ),
        ],
        outputs=[gr.DataFrame(label="Equations"), gr.Textbox(label="Error Log")],
    )
    # Add file to the demo:

    demo.launch()


if __name__ == "__main__":
    main()
