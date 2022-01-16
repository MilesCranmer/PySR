# Getting Started

## Installation
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

The second and additional calls of `pysr` will be significantly
faster in startup time, since the first call to Julia will compile
and cache functions from the symbolic regression backend.

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

