import copy
from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .export_jax import sympy2jax
from .export_numpy import sympy2numpy
from .export_sympy import create_sympy_symbols, pysr2sympy
from .export_torch import sympy2torch
from .utils import ArrayLike


def add_export_formats(
    output: pd.DataFrame,
    *,
    feature_names_in: ArrayLike[str],
    selection_mask: Union[NDArray[np.bool_], None] = None,
    extra_sympy_mappings: Optional[Dict[str, Callable]] = None,
    extra_torch_mappings: Optional[Dict[Callable, Callable]] = None,
    output_torch_format: bool = False,
    extra_jax_mappings: Optional[Dict[Callable, str]] = None,
    output_jax_format: bool = False,
) -> pd.DataFrame:

    output = copy.deepcopy(output)

    scores = []
    lastMSE = None
    lastComplexity = 0
    sympy_format = []
    lambda_format = []
    jax_format = []
    torch_format = []

    for _, eqn_row in output.iterrows():
        eqn = pysr2sympy(
            eqn_row["equation"],
            feature_names_in=feature_names_in,
            extra_sympy_mappings=extra_sympy_mappings,
        )
        sympy_format.append(eqn)

        # NumPy:
        sympy_symbols = create_sympy_symbols(feature_names_in)
        lambda_format.append(
            sympy2numpy(
                eqn,
                sympy_symbols,
                selection=selection_mask,
            )
        )

        # JAX:
        if output_jax_format:
            func, params = sympy2jax(
                eqn,
                sympy_symbols,
                selection=selection_mask,
                extra_jax_mappings=extra_jax_mappings,
            )
            jax_format.append({"callable": func, "parameters": params})

        # Torch:
        if output_torch_format:
            module = sympy2torch(
                eqn,
                sympy_symbols,
                selection=selection_mask,
                extra_torch_mappings=extra_torch_mappings,
            )
            torch_format.append(module)

        curMSE = eqn_row["loss"]
        curComplexity = eqn_row["complexity"]

        if lastMSE is None:
            cur_score = 0.0
        else:
            if curMSE > 0.0:
                # TODO Move this to more obvious function/file.
                cur_score = -np.log(curMSE / lastMSE) / (curComplexity - lastComplexity)
            else:
                cur_score = np.inf

        scores.append(cur_score)
        lastMSE = curMSE
        lastComplexity = curComplexity

    output["score"] = np.array(scores)
    output["sympy_format"] = sympy_format
    output["lambda_format"] = lambda_format
    output_cols = [
        "complexity",
        "loss",
        "score",
        "equation",
        "sympy_format",
        "lambda_format",
    ]
    if output_jax_format:
        output_cols += ["jax_format"]
        output["jax_format"] = jax_format
    if output_torch_format:
        output_cols += ["torch_format"]
        output["torch_format"] = torch_format

    return output[output_cols]
