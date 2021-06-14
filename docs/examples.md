# Examples

### Preamble

```python
import numpy as np
from pysr import *
```

We'll also set up some default options that will
make these simple searches go faster (but are less optimal
for more complex searches).

```python
kwargs = dict(populations=5, niterations=5, annealing=True)
```

1. Simple search

Here's a simple example where we turn off multiprocessing,
and find the expression `2 cos(x3) + x0^2 - 2`.

```python
X = 2 * np.random.randn(100, 5)
y = 2 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 2
expressions = pysr(X, y, binary_operators=["+", "-", "*", "/"], **kwargs)
print(best(expressions))
```

2. Custom operator

Here, we define a custom operator and use it to find an expression:

```python
X = 2 * np.random.randn(100, 5)
y = 1 / X[:, 0]
expressions = pysr(
    X,
    y,
    binary_operators=["plus", "mult"],
    unary_operators=["inv(x) = 1/x"],
    **kwargs
)
print(best(expressions))
```

3. Multiple outputs

Here, we do the same thing, but with multiple expressions at once,
each requiring a different feature.
```python
X = 2 * np.random.randn(100, 5)
y = 1 / X[:, [0, 1, 2]]
expressions = pysr(
    X,
    y,
    binary_operators=["plus", "mult"],
    unary_operators=["inv(x) = 1/x"],
    **kwargs
)
```

4. Plotting an expression

Here, let's use the same equations, but get a format we can actually
use and test. We can add this option after a search via the `get_hof`
function:

```python
expressions = get_hof(extra_sympy_mappings={"inv": lambda x: 1/x})
```
If you look at the lists of expressions before and after, you will
see that the sympy format now has replaced `inv` with `1/`.

For now, let's consider the expressions for output 0:
```python
expressions = expressions[0]
```
This is a pandas table, which we can filter:
```python
best_expression = expressions.iloc[expressions.MSE.argmin()]
```
We can see the LaTeX version of this with:
```python
import sympy
sympy.latex(best_expression.sympy_format)
```

We can access the numpy version with:
```python
f = best_expression.lambda_format
print(f)
```

Which shows a PySR object on numpy code:
```
>> PySRFunction(X=>1/x0)
```

Let's plot this against the truth:
```python
from matplotlib import pyplot as plt
plt.scatter(y[:, 0], f(X))
plt.xlabel('Truth')
plt.ylabel('Prediction')
plt.show()
```
Which gives us:
![](./images/example_plot.png)
