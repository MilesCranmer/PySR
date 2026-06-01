"""Checkpoint and pickle helpers for PySRRegressor."""

from __future__ import annotations

import logging
import os
import pickle as pkl
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import pandas as pd

    from .sr import PySRRegressor


logger = logging.getLogger(__name__)


def get_regressor_pickle_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return a pickle-safe version of a PySRRegressor state dictionary."""
    show_pickle_warning = not (
        "show_pickle_warnings_" in state and not state["show_pickle_warnings_"]
    )
    state_keys_to_clear = (
        "extra_sympy_mappings",
        "extra_jax_mappings",
        "extra_torch_mappings",
    )
    for state_key in state_keys_to_clear:
        warn_msg = (
            f"`{state_key}` cannot be pickled and will be removed from the "
            "serialized instance. When loading the model, please redefine "
            f"`{state_key}` at runtime."
        )
        if state[state_key] is not None:
            if show_pickle_warning:
                warnings.warn(warn_msg)
            else:
                logger.debug(warn_msg)
    state_keys_to_clear = (*state_keys_to_clear, "logger_")
    pickled_state = {
        key: (None if key in state_keys_to_clear else value)
        for key, value in state.items()
    }
    if ("equations_" in pickled_state) and (pickled_state["equations_"] is not None):
        pickled_state["output_torch_format"] = False
        pickled_state["output_jax_format"] = False
        pickled_state["equations_"] = drop_equation_columns(
            pickled_state["equations_"], ["jax_format", "torch_format"]
        )
        try:
            pkl.dumps(pickled_state["equations_"])
        except Exception as e:
            warn_msg = (
                "`equations_` export formats cannot be pickled and will be "
                "removed from the serialized instance. When loading the model, "
                "please redefine custom mappings at runtime."
            )
            if show_pickle_warning:
                warnings.warn(warn_msg)
            else:
                logger.debug(f"{warn_msg} Error: {e}")
            pickled_state["equations_"] = drop_equation_columns(
                pickled_state["equations_"], ["sympy_format", "lambda_format"]
            )
    return pickled_state


def drop_equation_columns(
    equations: pd.DataFrame | list[pd.DataFrame],
    columns: list[str],
) -> pd.DataFrame | list[pd.DataFrame]:
    if isinstance(equations, list):
        return [
            dataframe.loc[:, ~dataframe.columns.isin(columns)].copy()
            for dataframe in equations
        ]
    return equations.loc[:, ~equations.columns.isin(columns)].copy()


def equations_missing_export_formats(
    equations: pd.DataFrame | list[pd.DataFrame],
) -> bool:
    required_columns = {"sympy_format", "lambda_format"}
    if isinstance(equations, list):
        return any(
            not required_columns.issubset(dataframe.columns) for dataframe in equations
        )
    return not required_columns.issubset(equations.columns)


def save_checkpoint(model: PySRRegressor, pkl_filename: Path) -> None:
    tmp_filename = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=pkl_filename.parent, delete=False
        ) as f:
            tmp_filename = Path(f.name)
            pkl.dump(model, f)
        os.replace(tmp_filename, pkl_filename)
    except Exception as e:
        logger.debug(f"Error checkpointing model: {e}")
        if tmp_filename is not None:
            tmp_filename.unlink(missing_ok=True)


def load_checkpoint(pkl_filename: Path) -> PySRRegressor | None:
    if pkl_filename.stat().st_size == 0:
        logger.warning(
            f"Checkpoint file {pkl_filename} is empty. "
            "Attempting to recreate model from CSV backups..."
        )
        return None
    try:
        with open(pkl_filename, "rb") as f:
            return cast("PySRRegressor", pkl.load(f))
    except (EOFError, pkl.UnpicklingError) as e:
        logger.warning(
            f"Could not load checkpoint file {pkl_filename}: {e}. "
            "Attempting to recreate model from CSV backups..."
        )
        return None
