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

## 1. Simple search

Here's a simple example where we 
find the expression `2 cos(x3) + x0^2 - 2`.

```python
X = 2 * np.random.randn(100, 5)
y = 2 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 2
model = PySRRegressor(binary_operators=["+", "-", "*", "/"], **kwargs)
model.fit(X, y)
print(model)
```

## 2. Custom operator

Here, we define a custom operator and use it to find an expression:

```python
X = 2 * np.random.randn(100, 5)
y = 1 / X[:, 0]
model = PySRRegressor(
    binary_operators=["plus", "mult"],
    unary_operators=["inv(x) = 1/x"],
    **kwargs
)
model.fit(X, y)
print(model)
```

## 3. Multiple outputs

Here, we do the same thing, but with multiple expressions at once,
each requiring a different feature.
```python
X = 2 * np.random.randn(100, 5)
y = 1 / X[:, [0, 1, 2]]
model = PySRRegressor(
    binary_operators=["plus", "mult"],
    unary_operators=["inv(x) = 1/x"],
    **kwargs
)
model.fit(X, y)
```

## 4. Plotting an expression

Here, let's use the same equations, but get a format we can actually
use and test. We can add this option after a search via the `set_params`
function:

```python
model.set_params(extra_sympy_mappings={"inv": lambda x: 1/x})
model.sympy()
```
If you look at the lists of expressions before and after, you will
see that the sympy format now has replaced `inv` with `1/`.
We can again look at the equation chosen:
```python
print(model)
```

For now, let's consider the expressions for output 0.
We can see the LaTeX version of this with:
```python
model.latex()[0]
```
or output 1 with `model.latex()[1]`.


Let's plot the prediction against the truth:
```python
from matplotlib import pyplot as plt
plt.scatter(y[:, 0], model(X)[:, 0])
plt.xlabel('Truth')
plt.ylabel('Prediction')
plt.show()
```
Which gives us:

![](https://github.com/MilesCranmer/PySR/raw/master/docs/images/example_plot.png)

## 5. Additional features

For the many other features available in PySR, please
read the [Options section](options.md). 