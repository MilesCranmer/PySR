# [PySR](https://github.com/MilesCranmer/PySR)
**Parallelized symbolic regression built on Julia, and interfaced by Python.
Uses regularized evolution, simulated annealing, and gradient-free optimization.**
| **Docs** | **pip** |
|---|---|
|[![Documentation Status](https://readthedocs.org/projects/pysr/badge/?version=latest)](https://pysr.readthedocs.io/en/latest/?badge=latest)|[![PyPI version](https://badge.fury.io/py/pysr.svg)](https://badge.fury.io/py/pysr)|

(pronounced like *py* as in python, and then *sur* as in surface)

[Cite this software](https://github.com/MilesCranmer/PySR/blob/master/CITATION.md)


### Test status:
| **Linux** | **Windows** | **macOS** | **Coverage** | 
|---|---|---|---|
|[![.github/workflows/CI.yml](https://github.com/MilesCranmer/PySR/actions/workflows/CI.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI.yml)|[![.github/workflows/CI_Windows.yml](https://github.com/MilesCranmer/PySR/actions/workflows/CI_Windows.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI_Windows.yml)|[![CI_m](https://github.com/MilesCranmer/PySR/actions/workflows/CI_mac.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI_mac.yml)|[![Coverage Status](https://coveralls.io/repos/github/MilesCranmer/PySR/badge.svg?branch=master&service=github)](https://coveralls.io/github/MilesCranmer/PySR)|


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
pip install pysr
```

The first launch will automatically install the Julia packages
required. Most common issues at this stage are solved
by [tweaking the Julia package server](https://github.com/MilesCranmer/PySR/issues/27).
to use up-to-date packages.

# Quickstart

Here is some demo code (also found in `example.py`)
```python
import numpy as np
from pysr import pysr, best

# Dataset
X = 2 * np.random.randn(100, 5)
y = 2 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 2

# Learn equations
equations = pysr(
    X,
    y,
    niterations=5,
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",  # Pre-defined library of operators (see docs)
        "inv(x) = 1/x",  # Define your own operator! (Julia syntax)
    ],
)

...# (you can use ctl-c to exit early)

print(best(equations))
```

which gives:

```python
x0**2 + 2.000016*cos(x3) - 1.9999845
```

One can also use `best_tex` to get the LaTeX form,
or `best_callable` to get a function you can call.
This uses a score which balances complexity and error;
however, one can see the full list of equations with:
```python
print(equations)
```
This is a pandas table, with additional columns:

- `MSE` - the mean square error of the formula
- `score` - a metric akin to Occam's razor; you should use this to help select the "true" equation.
- `sympy_format` - sympy equation.
- `lambda_format` - a lambda function for that equation, that you can pass values through.
