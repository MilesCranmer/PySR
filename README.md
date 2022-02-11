[//]: # (Logo:)

<img src="https://raw.githubusercontent.com/MilesCranmer/PySR/master/docs/images/pysr_logo.svg" width="400" />

# PySR: High-Performance Symbolic Regression in Python

PySR is built on an extremely optimized pure-Julia backend, and uses regularized evolution, simulated annealing, and gradient-free optimization to search for equations that fit your data.

| **Docs** | **pip** | **conda** | **Stats** |
|---|---|---|---|
|[![Documentation](https://github.com/MilesCranmer/PySR/actions/workflows/docs.yml/badge.svg)](https://astroautomata.com/PySR/)|[![PyPI version](https://badge.fury.io/py/pysr.svg)](https://badge.fury.io/py/pysr)|[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pysr.svg)](https://anaconda.org/conda-forge/pysr)|[![Downloads](https://pepy.tech/badge/pysr)](https://badge.fury.io/py/pysr)|


(pronounced like *py* as in python, and then *sur* as in surface)

If you find PySR useful, please cite it using the citation information given in [CITATION.md](https://github.com/MilesCranmer/PySR/blob/master/CITATION.md).
If you've finished a project with PySR, please let me know and I may showcase your work here!


### Test status:
| **Linux** | **Windows** | **macOS (intel)** |
|---|---|---|
|[![Linux](https://github.com/MilesCranmer/PySR/actions/workflows/CI.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI.yml)|[![Windows](https://github.com/MilesCranmer/PySR/actions/workflows/CI_Windows.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI_Windows.yml)|[![macOS](https://github.com/MilesCranmer/PySR/actions/workflows/CI_mac.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI_mac.yml)|
| **Docker** | **Conda** | **Coverage** | 
|[![Docker](https://github.com/MilesCranmer/PySR/actions/workflows/CI_docker.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI_docker.yml)|[![conda-forge](https://github.com/MilesCranmer/PySR/actions/workflows/CI_conda_forge.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI_conda_forge.yml)|[![Coverage Status](https://coveralls.io/repos/github/MilesCranmer/PySR/badge.svg?branch=master&service=github)](https://coveralls.io/github/MilesCranmer/PySR)|


Check out [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl) for
the pure-Julia backend of this package.

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
PySR uses both Julia and Python, so you need to have both installed
(or use the conda-forge one, which does this for you).
To install Julia - see the [Julia website](https://julialang.org/downloads/).

You can install PySR with pip:
```bash
pip3 install pysr
```
or conda (which also installs julia for you):
```bash
conda install -c conda-forge pysr
```
Both of these should be followed by the following setup command:
```bash
python3 -c 'import pysr; pysr.install()'
```
This will install and update the required Julia packages, including
`PyCall.jl`.


Most common issues at this stage are solved
by [tweaking the Julia package server](https://github.com/MilesCranmer/PySR/issues/27)
to use up-to-date packages.

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
    niterations=5,
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",  # Custom operator (julia syntax)
    ],
    model_selection="best",
    loss="loss(x, y) = (x - y)^2",  # Custom loss function (julia syntax)
)
```
This will set up the model for 5 iterations of the search code, which contains hundreds of thousands of mutations and equation evaluations.

Let's train this model on our dataset:
```python
model.fit(X, y)
```
Internally, this launches a Julia process which will do a multithreaded search for equations to fit the dataset.

Equations will be printed during training, and once you are satisfied, you may 
quit early by hitting 'q' and then \<enter\>.

After the model has been fit, you can run `model.predict(X)`
to see the predictions on a given dataset.

You may run:
```python
print(model)
```
to print the learned equations:
```python
PySRRegressor.equations = [
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

`model.equations` is a pandas DataFrame containing all equations, including callable format 
(`lambda_format`),
SymPy format (`sympy_format`), and even JAX and PyTorch format 
(both of which are differentiable).

Note that `PySRRegressor` stores the state of the last search, and will restart from where you left off the next time you call `.fit()`. This will cause problems if significant changes are made to the search parameters (like changing the operators). You can run `model.reset()` to reset the state.

There are several other useful features such as denoising (e.g., `denoising=True`),
feature selection (e.g., `select_k_features=3`).
For examples of these and other features, see the [examples page](https://astroautomata.com/PySR/#/examples).
For a detailed look at more options, see the [options page](https://astroautomata.com/PySR/#/options).
You can also see the full API at [this page](https://astroautomata.com/PySR/#/api).


# Docker

You can also test out PySR in Docker, without
installing it locally, by running the following command in
the root directory of this repo:
```bash
docker build --pull --rm -f "Dockerfile" -t pysr "."
```
This builds an image called `pysr`. If you have issues building (for example, on Apple Silicon),
you can emulate an architecture that works by including: `--platform linux/amd64`.
You can then run this with:
```bash
docker run -it --rm -v "$PWD:/data" pysr ipython
```
which will link the current directory to the container's `/data` directory
and then launch ipython.
