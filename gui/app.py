import io
import gradio as gr
import sys
import os
import tempfile
import numpy as np
import pandas as pd
import traceback as tb

empty_df = pd.DataFrame(
    {
        "equation": [],
        "loss": [],
        "complexity": [],
    }
)
Main = None

def greet(
    file_obj: tempfile._TemporaryFileWrapper,
    col_to_fit: str,
    niterations: int,
    binary_operators: list,
    unary_operators: list,
):
    global Main
    if Main is not None:
        return (
            empty_df,
            "Refresh the page to run with a different configuration."
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

    # Install Julia:
    os.system(
        """if [ ! -d "/home/user/julia" ]; then
        wget https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.3-linux-x86_64.tar.gz
        tar zxvf julia-1.7.3-linux-x86_64.tar.gz
        mkdir /home/user/julia
        mv julia-1.7.3/* /home/user/julia/
    fi""")
    os.environ["PATH"] += ":/home/user/julia/bin/"
    # Need to install PySR in separate python instance:
    os.system(
        """if [ ! -d "/home/user/.julia/environments/pysr-0.9.3" ]; then
        export PATH="$PATH:/home/user/julia/bin/"
        python -c 'import pysr; pysr.install()'
    fi"""
    )

    import pysr
    try:
        from julia.api import JuliaInfo
        info = JuliaInfo.load(julia="/home/user/julia/bin/julia")
        from julia import Main as _Main
        pysr.sr.Main = _Main
    except Exception as e:
        error_message = tb.format_exc()
        return (
            empty_df,
            error_message,
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
    try:
        model.fit(X, y)
    # Catch all error:
    except Exception as e:
        error_traceback = tb.format_exc()
        if "CalledProcessError" in error_traceback:
            return (
                empty_df,
                "Could not initialize Julia. Error message:\n"
                + error_traceback,
            )
        else:
            return (
                empty_df,
                "Failed due to error:\n" + error_traceback,
            )

    df = model.equations_[["equation", "loss", "complexity"]]
    # Convert all columns to string type:
    df = df.astype(str)
    return df, "Successful."


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
