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
from pysr import pysr

# Dataset
X = 2*np.random.randn(100, 5)
y = 2*np.cos(X[:, 3]) + X[:, 0]**2 - 2

# Learn equations
equations = pysr(X, y, niterations=5,
            binary_operators=["plus", "mult"],
            unary_operators=["cos", "exp", "sin"])

...

print(equations)
```

which gives:

```
   Complexity       MSE                                                Equation
0           5  1.947431                          plus(-1.7420927, mult(x0, x0))
1           8  0.486858           plus(-1.8710494, plus(cos(x3), mult(x0, x0)))
2          11  0.000000  plus(plus(mult(x0, x0), cos(x3)), plus(-2.0, cos(x3)))
```

The newest version of PySR also returns three additional columns:

- `score` - a metric akin to Occam's razor; you should use this to help select the "true" equation.
- `sympy_format` - sympy equation.
- `lambda_format` - a lambda function for that equation, that you can pass values through.

