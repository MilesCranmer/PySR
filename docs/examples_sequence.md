# Toy Sequence Examples with Code

## Preamble
```python
import numpy as np
from pysr import *
```

Note that most of the functionality
of PySRSequenceRegressor is inherited
from PySRRegressor.

## 1. Simple Search

Here's a simple example where we
find the expression `f(n) = f(n-1) + f(n-2)`.

```python
X = [1, 1]
for i in range(20):
    X.append(X[-1] + X[-2])
X = np.array(X)
model = PySRSequenceRegressor(
    recursive_history_length=2,
    binary_operators=["+", "-", "*", "/"]
)
model.fit(X)  # no y needed
print(model)
```

## 2. Multidimensionality

Here we find a 2D recurrence relation
with two data points at a time.

```python
X = [[1, 2], [3, 4]]
for i in range(100):
    X.append([
        X[-1][0] + X[-2][0],
        X[-1][1] - X[-2][1]
    ])
X = np.array(X)

model = PySRSequenceRegressor(
    recursive_history_length=2,
    binary_operators=["+", "*"],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
)

model.fit(X)
print(model)
```
