"""Functions for doing feature selection during preprocessing."""

from typing import Optional, cast

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from .utils import ArrayLike


def run_feature_selection(
    X: ndarray,
    y: ndarray,
    select_k_features: int,
    random_state: Optional[np.random.RandomState] = None,
) -> NDArray[np.bool_]:
    """
    Find most important features.

    Uses a gradient boosting tree regressor as a proxy for finding
    the k most important features in X, returning indices for those
    features as output.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel

    clf = RandomForestRegressor(
        n_estimators=100, max_depth=3, random_state=random_state
    )
    clf.fit(X, y)
    selector = SelectFromModel(
        clf, threshold=-np.inf, max_features=select_k_features, prefit=True
    )
    return cast(NDArray[np.bool_], selector.get_support(indices=False))


# Function has not been removed only due to usage in module tests
def _handle_feature_selection(
    X: ndarray,
    select_k_features: Optional[int],
    y: ndarray,
    variable_names: ArrayLike[str],
):
    if select_k_features is not None:
        selection = run_feature_selection(X, y, select_k_features)
        print(f"Using features {[variable_names[i] for i in selection]}")
        X = X[:, selection]
    else:
        selection = None

    return X, selection
