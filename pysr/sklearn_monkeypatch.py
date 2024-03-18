# Here, we monkey patch scikit-learn until this
# issue is fixed: https://github.com/scikit-learn/scikit-learn/issues/25922
from sklearn.utils import validation


def _ensure_no_complex_data(*args, **kwargs): ...


try:
    validation._ensure_no_complex_data = _ensure_no_complex_data
except AttributeError:  # pragma: no cover
    ...
