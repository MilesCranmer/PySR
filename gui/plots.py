import matplotlib

matplotlib.use("agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.ioff()
plt.rcParams["font.family"] = "monospace"
# plt.rcParams["font.family"] = [
#     "IBM Plex Mono",
#     # Fallback fonts:
#     "DejaVu Sans Mono",
#     "Courier New",
#     "monospace",
# ]

from data import generate_data


def plot_pareto_curve(df: pd.DataFrame, maxsize: int):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    if len(df) == 0 or "Equation" not in df.columns:
        return fig

    ax.loglog(
        df["Complexity"],
        df["Loss"],
        marker="o",
        linestyle="-",
        color="#333f48",
        linewidth=1.5,
        markersize=6,
    )

    ax.set_xlim(0.5, maxsize + 1)
    ytop = 2 ** (np.ceil(np.log2(df["Loss"].max())))
    ybottom = 2 ** (np.floor(np.log2(df["Loss"].min() + 1e-20)))
    ax.set_ylim(ybottom, ytop)

    stylize_axis(ax)

    ax.set_xlabel("Complexity")
    ax.set_ylabel("Loss")
    fig.tight_layout(pad=2)

    return fig


def plot_example_data(test_equation, num_points, noise_level, data_seed):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    X, y = generate_data(test_equation, num_points, noise_level, data_seed)
    x = X["x"]

    ax.scatter(x, y, alpha=0.7, edgecolors="w", s=50)

    stylize_axis(ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout(pad=2)

    return fig


def plot_predictions(y, ypred):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    ax.scatter(y, ypred, alpha=0.7, edgecolors="w", s=50)

    stylize_axis(ax)

    ax.set_xlabel("true")
    ax.set_ylabel("prediction")
    fig.tight_layout(pad=2)

    return fig


def stylize_axis(ax):
    ax.grid(True, which="both", ls="--", linewidth=0.5, color="gray", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Range-frame the plot
    for direction in ["bottom", "left"]:
        ax.spines[direction].set_position(("outward", 10))

    # Delete far ticks
    ax.tick_params(axis="both", which="major", labelsize=10, direction="out", length=5)
    ax.tick_params(axis="both", which="minor", labelsize=8, direction="out", length=3)
