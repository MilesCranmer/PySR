import io
import gradio as gr
import os
import tempfile
from typing import List


def greet(file_obj: List[tempfile._TemporaryFileWrapper]):
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

    df = pd.read_csv(file_obj[0])
    # y = np.array(df["y"])
    # X = df.drop(["y"], axis=1)

    # model = PySRRegressor(update=False, temp_equation_file=True)
    # model.fit(X, y)

    # df_output = model.equations_
    df_output = df
    df_output.to_csv("output.csv", index=False, sep="\t")

    return "output.csv"


demo = gr.Interface(
    fn=greet,
    description="A demo of PySR",
    inputs=gr.File(label="Upload a CSV file", file_count=1),
    outputs=gr.File(label="Equation List"),
)
# Add file to the demo:

demo.launch()
