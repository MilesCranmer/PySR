[//]: # (Logo:)

<div align="center">

https://user-images.githubusercontent.com/7593028/188328887-1b6cda72-2f41-439e-ae23-75dd3489bc24.mp4

# PySR: High-Performance Symbolic Regression in Python

</div>


PySR uses evolutionary algorithms to search for symbolic expressions which optimize a particular objective.

<div align="center">

| **Docs** | **colab** | **pip** | **conda** | **Stats** |
|---|---|---|---|---|
|[![Documentation](https://github.com/MilesCranmer/PySR/actions/workflows/docs.yml/badge.svg)](https://astroautomata.com/PySR/)|[![Colab](https://img.shields.io/badge/colab-notebook-yellow)](https://colab.research.google.com/github/MilesCranmer/PySR/blob/master/examples/pysr_demo.ipynb)|[![PyPI version](https://badge.fury.io/py/pysr.svg)](https://badge.fury.io/py/pysr)|[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pysr.svg)](https://anaconda.org/conda-forge/pysr)|<div align="center">pip: [![Downloads](https://pepy.tech/badge/pysr)](https://badge.fury.io/py/pysr)<br>conda: [![Anaconda-Server Badge](https://anaconda.org/conda-forge/pysr/badges/downloads.svg)](https://anaconda.org/conda-forge/pysr)</div>|

</div>


(pronounced like *py* as in python, and then *sur* as in surface)

If you find PySR useful, please cite it using the citation information given in [CITATION.md](https://github.com/MilesCranmer/PySR/blob/master/CITATION.md).
If you've finished a project with PySR, please submit a PR to showcase your work on the [Research Showcase page](https://astroautomata.com/PySR/papers)!


<div align="center">

### Test status

| **Linux** | **Windows** | **macOS (intel)** |
|---|---|---|
|[![Linux](https://github.com/MilesCranmer/PySR/actions/workflows/CI.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI.yml)|[![Windows](https://github.com/MilesCranmer/PySR/actions/workflows/CI_Windows.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI_Windows.yml)|[![macOS](https://github.com/MilesCranmer/PySR/actions/workflows/CI_mac.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI_mac.yml)|
| **Docker** | **Conda** | **Coverage** | 
|[![Docker](https://github.com/MilesCranmer/PySR/actions/workflows/CI_docker.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI_docker.yml)|[![conda-forge](https://github.com/MilesCranmer/PySR/actions/workflows/CI_conda_forge.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI_conda_forge.yml)|[![Coverage Status](https://coveralls.io/repos/github/MilesCranmer/PySR/badge.svg?branch=master&service=github)](https://coveralls.io/github/MilesCranmer/PySR)|


</div>

PySR is built on an extremely optimized pure-Julia backend: [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl).

Symbolic regression is a very interpretable machine learning algorithm
for low-dimensional problems: these tools search equation space
to find algebraic relations that approximate a dataset.

One can also
extend these approaches to higher-dimensional
spaces by using a neural network as proxy, as explained in 
[2006.11287](https://arxiv.org/abs/2006.11287), where we apply
it to N-body problems. Here, one essentially uses
symbolic regression to convert a neural net
to an analytic equation. Thus, these tools simultaneously present
an explicit and powerful way to interpret deep models.


*Backstory:*

Previously, we have used
[eureqa](https://www.creativemachineslab.com/eureqa.html),
which is a very efficient and user-friendly tool. However,
eureqa is GUI-only, doesn't allow for user-defined
operators, has no distributed capabilities,
and has become proprietary (and recently been merged into an online
service). Thus, the goal
of this package is to have an open-source symbolic regression tool
as efficient as eureqa, while also exposing a configurable
python interface.


# Installation

<div align="center">

| pip - **recommended** <br> (works everywhere) | conda <br>(Linux and Intel-based macOS) | docker <br>(if all else fails) |
|---|---|---|
| 1. [Install Julia](https://julialang.org/downloads/)<br>2. Then, run: `pip install -U pysr`<br>3. Finally, to install Julia packages:<br>`python -c 'import pysr; pysr.install()'` | `conda install -c conda-forge pysr` | 1. Clone this repo.<br>2. `docker build -t pysr .`<br>Run with:<br>`docker run -it --rm pysr ipython`

</div>

Common issues tend to be related to Python not finding Julia.
To debug this, try running `python -c 'import os; print(os.environ["PATH"])'`.
If none of these folders contain your Julia binary, then you need to add Julia's `bin` folder to your `PATH` environment variable.

**Running PySR on macOS with an M1 processor:** you should use the pip version, and make sure to get the Julia binary for ARM/M-series processors.

# Introduction

Let's create a PySR example. First, let's import
numpy to generate some test data:

```python
import numpy as np

X = 2 * np.random.randn(100, 5)
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5
```

We have created a dataset with 100 datapoints, with 5 features each.
The relation we wish to model is $2.5382 \cos(x_3) + x_0^2 - 0.5$.

Now, let's create a PySR model and train it.
PySR's main interface is in the style of scikit-learn:

```python
from pysr import PySRRegressor

model = PySRRegressor(
    niterations=40,  # < Increase me for better results
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
)
```

This will set up the model for 40 iterations of the search code, which contains hundreds of thousands of mutations and equation evaluations.

Let's train this model on our dataset:

```python
model.fit(X, y)
```

Internally, this launches a Julia process which will do a multithreaded search for equations to fit the dataset.

Equations will be printed during training, and once you are satisfied, you may 
quit early by hitting 'q' and then \<enter\>.

After the model has been fit, you can run `model.predict(X)`
to see the predictions on a given dataset using the automatically-selected expression,
or, for example, `model.predict(X, 3)` to see the predictions of the 3rd equation.

You may run:

```python
print(model)
```

to print the learned equations:

```python
PySRRegressor.equations_ = [
	   pick     score                                           equation       loss  complexity
	0        0.000000                                          4.4324794  42.354317           1
	1        1.255691                                          (x0 * x0)   3.437307           3
	2        0.011629                          ((x0 * x0) + -0.28087974)   3.358285           5
	3        0.897855                              ((x0 * x0) + cos(x3))   1.368308           6
	4        0.857018                ((x0 * x0) + (cos(x3) * 2.4566472))   0.246483           8
	5  >>>>       inf  (((cos(x3) + -0.19699033) * 2.5382123) + (x0 *...   0.000000          10
]
```

This arrow in the `pick` column indicates which equation is currently selected by your
`model_selection` strategy for prediction.
(You may change `model_selection` after `.fit(X, y)` as well.)

`model.equations_` is a pandas DataFrame containing all equations, including callable format 
(`lambda_format`),
SymPy format (`sympy_format` - which you can also get with `model.sympy()`), and even JAX and PyTorch format 
(both of which are differentiable - which you can get with `model.jax()` and `model.pytorch()`).

Note that `PySRRegressor` stores the state of the last search, and will restart from where you left off the next time you call `.fit()`, assuming you have set `warm_start=True`.
This will cause problems if significant changes are made to the search parameters (like changing the operators). You can run `model.reset()` to reset the state.

You will notice that PySR will save two files: `hall_of_fame...csv` and `hall_of_fame...pkl`.
The csv file is a list of equations and their losses, and the pkl file is a saved state of the model.
You may load the model from the `pkl` file with:

```python
model = PySRRegressor.from_file("hall_of_fame.2022-08-10_100832.281.pkl")
``` 

There are several other useful features such as denoising (e.g., `denoising=True`),
feature selection (e.g., `select_k_features=3`).
For examples of these and other features, see the [examples page](https://astroautomata.com/PySR/examples).
For a detailed look at more options, see the [options page](https://astroautomata.com/PySR/options).
You can also see the full API at [this page](https://astroautomata.com/PySR/api).

## Detailed Example

The following code makes use of as many PySR features as possible.
Note that is just a demonstration of features and you should not use this example as-is.
For details on what each parameter does, check out the [API page](https://astroautomata.com/PySR/api/).

```python
model = PySRRegressor(
    procs=4,
    populations=8,
    # ^ 2 populations per core, so one is always running.
    population_size=50,
    # ^ Slightly larger populations, for greater diversity.
    ncyclesperiteration=500, 
    # ^ Generations between migrations.
    niterations=10000000,  # Run forever
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"
        # Stop early if we find a good and simple equation
    ),
    timeout_in_seconds=60 * 60 * 24,
    # ^ Alternatively, stop after 24 hours have passed.
    maxsize=50,
    # ^ Allow greater complexity.
    maxdepth=10,
    # ^ But, avoid deep nesting.
    binary_operators=["*", "+", "-", "/"],
    unary_operators=["square", "cube", "exp", "cos2(x)=cos(x)^2"],
    constraints={
        "/": (-1, 9),
        "square": 9,
        "cube": 9,
        "exp": 9,
    },
    # ^ Limit the complexity within each argument.
    # "inv": (-1, 9) states that the numerator has no constraint,
    # but the denominator has a max complexity of 9.
    # "exp": 9 simply states that `exp` can only have
    # an expression of complexity 9 as input.
    nested_constraints={
        "square": {"square": 1, "cube": 1, "exp": 0},
        "cube": {"square": 1, "cube": 1, "exp": 0},
        "exp": {"square": 1, "cube": 1, "exp": 0},
    },
    # ^ Nesting constraints on operators. For example,
    # "square(exp(x))" is not allowed, since "square": {"exp": 0}.
    complexity_of_operators={"/": 2, "exp": 3},
    # ^ Custom complexity of particular operators.
    complexity_of_constants=2,
    # ^ Punish constants more than variables
    select_k_features=4,
    # ^ Train on only the 4 most important features
    progress=True,
    # ^ Can set to false if printing to a file.
    weight_randomize=0.1,
    # ^ Randomize the tree much more frequently
    cluster_manager=None,
    # ^ Can be set to, e.g., "slurm", to run a slurm
    # cluster. Just launch one script from the head node.
    precision=64,
    # ^ Higher precision calculations.
    warm_start=True,
    # ^ Start from where left off.
    turbo=True,
    # ^ Faster evaluation (experimental)
    julia_project=None,
    # ^ Can set to the path of a folder containing the
    # "SymbolicRegression.jl" repo, for custom modifications.
    update=False,
    # ^ Don't update Julia packages
    extra_sympy_mappings={"cos2": lambda x: sympy.cos(x)**2},
    # extra_torch_mappings={sympy.cos: torch.cos},
    # ^ Not needed as cos already defined, but this
    # is how you define custom torch operators.
    # extra_jax_mappings={sympy.cos: "jnp.cos"},
    # ^ For JAX, one passes a string.
)
```

# Docker

You can also test out PySR in Docker, without
installing it locally, by running the following command in
the root directory of this repo:

```bash
docker build -t pysr .
```

This builds an image called `pysr` for your system's architecture,
which also contains IPython.

You can then run this with:

```bash
docker run -it --rm -v "$PWD:/data" pysr ipython
```

which will link the current directory to the container's `/data` directory
and then launch ipython.

If you have issues building for your system's architecture,
you can emulate another architecture by including `--platform linux/amd64`,
before the `build` and `run` commands.
