import numpy as np

from pysr import PySRSequenceRegressor

X = np.asarray([[1, 2], [3, 4], [4, 6], [7, 10], [11, 16], [18, 26]])
model = PySRSequenceRegressor(
    recursive_history_length=3,
    niterations=20
)
model.fit(X)
print(model._regressor.__dict__)
print(model._regressor.__repr__())
print(hasattr(model._regressor, 'feature_names_in_'))
print(hasattr(model._regressor, 'selection_mask_'))
print(hasattr(model._regressor, 'nout_'))

print(X.shape)
pred = model.predict(X, extra_predictions=3)
print(pred)
print(pred.shape)

print(model.equations_)
print(model.latex())