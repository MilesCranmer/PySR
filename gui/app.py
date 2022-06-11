import streamlit as st
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
    import pysr

    pysr.install()
    from pysr import PySRRegressor

    model = PySRRegressor()
    model.fit(X, y)

    col1.header("Equation")
    col2.header("Loss")
    # model.equations_ is a pd.DataFrame
    for i, row in model.equations_.iterrows():
        col1.subheader(row["equation"])
        col2.subheader(row["loss"])
