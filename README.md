[//]: # (Logo:)

<img src="https://raw.githubusercontent.com/MilesCranmer/PySR/master/pysr_logo.svg" width="400" />

**PySR: parallel symbolic regression built on Julia, and interfaced by Python.**

Uses regularized evolution, simulated annealing, and gradient-free optimization.

| **Docs** | **pip** |
|---|---|
|[![Documentation Status](https://readthedocs.org/projects/pysr/badge/?version=latest)](https://pysr.readthedocs.io/en/latest/?badge=latest)|[![PyPI version](https://badge.fury.io/py/pysr.svg)](https://badge.fury.io/py/pysr)|

(pronounced like *py* as in python, and then *sur* as in surface)

[Cite this software](https://github.com/MilesCranmer/PySR/blob/master/CITATION.md)


### Test status:
| **Linux** | **Windows** | **macOS** | **Docker** | **Coverage** | 
|---|---|---|---|---|
|[![Linux](https://github.com/MilesCranmer/PySR/actions/workflows/CI.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI.yml)|[![Windows](https://github.com/MilesCranmer/PySR/actions/workflows/CI_Windows.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI_Windows.yml)|[![macOS](https://github.com/MilesCranmer/PySR/actions/workflows/CI_mac.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI_mac.yml)|[![Docker](https://github.com/MilesCranmer/PySR/actions/workflows/CI_docker.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI_docker.yml)|[![Coverage Status](https://coveralls.io/repos/github/MilesCranmer/PySR/badge.svg?branch=master&service=github)](https://coveralls.io/github/MilesCranmer/PySR)|


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
PySR uses both Julia and Python, so you need to have both installed.

Install Julia - see [downloads](https://julialang.org/downloads/), and
then instructions for [mac](https://julialang.org/downloads/platform/#macos)
and [linux](https://julialang.org/downloads/platform/#linux_and_freebsd).
(Don't use the `conda-forge` version; it doesn't seem to work properly.)

You can install PySR with:
```bash
pip3 install pysr
python3 -c 'import pysr; pysr.install()'
```
The second line will install and update the required Julia packages, including
`PyCall.jl`.


Most common issues at this stage are solved
by [tweaking the Julia package server](https://github.com/MilesCranmer/PySR/issues/27).
to use up-to-date packages.

# Quickstart

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
    populations=8,
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
    ],
    model_selection="best",
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
to print the learned equations, which for the above should be close to:
```python
PySRRegressor.equations = [
   pick      score                                           Equation           MSE  Complexity
0         0.000000                                           3.598587  3.044337e+01           1
1         1.074135                                          (x0 * x0)  3.552313e+00           3
2         0.023611                          (-0.40477127 + (x0 * x0))  3.388464e+00           5
3         0.855682                              ((x0 * x0) + cos(x3))  1.440074e+00           6
4         0.876831                ((x0 * x0) + (2.5026207 * cos(x3)))  2.493328e-01           8
5  >>>>  10.687394  ((-0.5000114 + (x0 * x0)) + (2.5382013 * cos(x...  1.299652e-10          10
6         2.573098  ((-0.50000024 + (x0 * x0)) + (2.5382 * sin(1.5...  7.565937e-13          12
]
```
This arrow in the `pick` column indicates which equation is currently selected by your
`model_selection` strategy for prediction.
(You may change `model_selection` after `.fit(X, y)` as well.)

`model.equations` is a pandas DataFrame containing all equations, including callable format 
(`lambda_format`),
SymPy format (`sympy_format`), and even JAX and PyTorch format 
(both of which are differentiable).


### Notes

- `score` - a metric akin to Occam's razor; you should use this to help select the "true" equation.
- `sympy_format` - sympy equation.
- `lambda_format` - a lambda function for that equation, that you can pass values through.


# Docker

You can also test out PySR in Docker, without
installing it locally, by running the following command in
the root directory of this repo:
```bash
docker build --pull --rm -f "Dockerfile" -t pysr "."
```
This builds an image called `pysr`. You can then run this with:
```bash
docker run -it --rm -v "$PWD:/data" pysr ipython
```
which will link the current directory to the container's `/data` directory
and then launch ipython.
