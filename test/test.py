import numpy as np
from pysr import pysr
X = np.random.randn(100, 5)
y = X[:, 0]

equations = pysr(X, y)
print(equations)
