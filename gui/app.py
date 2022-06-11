import streamlit as st
import os

# Need to install PySR in separate python instance:
os.system(
    """
if [ ! -d "$HOME/.julia/environments/pysr-0.9.1" ]; then
    python -c 'import pysr; pysr.install()'
fi
"""
)
import pysr
from pysr import PySRRegressor
import numpy as np
import pandas as pd

st.title("Interactive PySR")
file_name = st.file_uploader(
    "Upload a data file, with your output column labeled 'y'", type=["csv"]
)

if file_name is not None:
    col1, col2 = st.columns(2)

    df = pd.read_csv(file_name)
    y = np.array(df["y"])
    X = df.drop(["y"], axis=1)

    model = PySRRegressor(update=False)
    model.fit(X, y)

    col1.header("Equation")
    col2.header("Loss")
    for i, row in model.equations_.iterrows():
        col1.subheader(str(row["equation"]))
        col2.subheader(str(row["loss"]))

    model = None

Main = None
pysr.sr.Main = None
pysr.sr.already_ran = False
