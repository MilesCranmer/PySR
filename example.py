import numpy as np
from pysr import PySRSequenceRegressor

X = np.asarray([[1, 2], [3, 4], [4, 6], [7, 10], [11, 16], [18, 26]])
model = PySRSequenceRegressor(recursive_history_length=2)
model.fit(X)

print(X.shape)
pred = model.predict(X)
print(pred)
print(pred.shape)
