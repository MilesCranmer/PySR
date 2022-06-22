import io
import gradio as gr
import os
import tempfile


def greet(
    file_obj: tempfile._TemporaryFileWrapper,
    col_to_fit: str,
    niterations: int,
    binary_operators: list,
    unary_operators: list,
):
    if col_to_fit == "":
        raise ValueError("Please enter a column to predict")
    niterations = int(niterations)
    # Need to install PySR in separate python instance:
    os.system(
        """if [ ! -d "$HOME/.julia/environments/pysr-0.9.1" ]
    then
        python -c 'import pysr; pysr.install()'
    fi"""
    )
    from pysr import PySRRegressor
    import numpy as np
    import pandas as pd

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

    return model.equations_


def main():
    demo = gr.Interface(
        fn=greet,
        description="A demo of PySR",
        inputs=[
            gr.File(label="Upload a CSV file"),
            gr.Textbox(placeholder="Column to predict"),
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
        outputs="dataframe",
    )
    # Add file to the demo:

    demo.launch()


if __name__ == "__main__":
    main()
