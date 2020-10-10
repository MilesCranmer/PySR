# Getting Started

## Installation
PySR uses both Julia and Python, so you need to have both installed.

Install Julia - see [downloads](https://julialang.org/downloads/), and
then instructions for [mac](https://julialang.org/downloads/platform/#macos)
and [linux](https://julialang.org/downloads/platform/#linux_and_freebsd).
(Don't use the `conda-forge` version; it doesn't seem to work properly.)
Then, at the command line,
install the `Optim` and `SpecialFunctions` packages via:

```bash
julia -e 'import Pkg; Pkg.add("Optim"); Pkg.add("SpecialFunctions")'
```

For python, you need to have Python 3, numpy, sympy, and pandas installed.

You can install this package from PyPI with:

```bash
pip install pysr
```

## Quickstart

```python
import numpy as np
from pysr import pysr, best, get_hof

# Dataset
X = 2*np.random.randn(100, 5)
y = 2*np.cos(X[:, 3]) + X[:, 0]**2 - 2

# Learn equations
equations = pysr(X, y, niterations=5,
        binary_operators=["plus", "mult"],
        unary_operators=["cos", "exp", "sin"])

...# (you can use ctl-c to exit early)

print(best())
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
print(get_hof())
```
This is a pandas table, with additional columns:

- `MSE` - the mean square error of the formula
- `score` - a metric akin to Occam's razor; you should use this to help select the "true" equation.
- `sympy_format` - sympy equation.
- `lambda_format` - a lambda function for that equation, that you can pass values through.

