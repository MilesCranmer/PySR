"""Wrapper around PyTorch modules to run PySR on them."""
import numpy as np

from ..sr import PySRRegressor
from ..julia_helpers import init_julia


def _create_pysr_wrapper(*args, **kwargs):
    """Julia must be initialized *before* pytorch is imported."""
    init_julia()

    import torch
    from torch import nn

    class _PySRWrapper(nn.Module):
        def __init__(
            self,
            model: nn.Module,
            max_observations: int = 1000,
            record: bool = True,
        ):
            super().__init__()
            self.model = model
            self.observations = []
            self.max_observations = max_observations
            self.record = record

            self.regressor = None
            self._is_recording = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Pass through model, potentially recording observations."""
            if self._is_recording:
                tensor_data_x = x.detach().cpu().numpy()

            y = self.model(x)

            if self._is_recording:
                tensor_data_y = y.detach().cpu().numpy()

                self.observations.append((tensor_data_x, tensor_data_y))
                if len(self.observations) > self.max_observations:
                    self.observations.pop(0)

            return y

        def start_recording(self):
            """Start recording observations of all PySRWrapper modules."""
            if self.record:
                self._is_recording = True
            for _, module in self.named_modules():
                if isinstance(module, _PySRWrapper):
                    if module.record:
                        module._is_recording = True

        def stop_recording(self):
            """Stop recording observations of all PySRWrapper modules."""
            self._is_recording = False
            for _, module in self.named_modules():
                if isinstance(module, _PySRWrapper):
                    module._is_recording = False

        def distill(
            self,
            regressor: PySRRegressor = None,
            **regressor_kwargs,
        ):
            """
            Fit a symbolic regression model to the observations.

            Parameters
            ----------
            regressor : PySRRegressor
                A PySRRegressor instance. If `None`, a new one will be created
                using the keyword arguments `regressor_kwargs`.
            **regressor_kwargs: dict
                Keyword arguments to pass to the PySRRegressor, if a new one
                is created.

            Returns
            -------
            regressor : PySRRegressor
                The fitted regressor.
            """
            if regressor is None:
                regressor = PySRRegressor(**regressor_kwargs)

            self.regressor = regressor

            X = np.concatenate([x for x, _ in self.observations], axis=0)
            y = np.concatenate([y for _, y in self.observations], axis=0)

            self.regressor.fit(X, y)

            return self.regressor

    return _PySRWrapper(*args, **kwargs)


def PySRWrapper(model, max_observations: int = 100, record: bool = True):
    """
    A wrapper for PyTorch that fits symbolic regression to specific modules.

    To begin recording observations of the model,
    call `.start_recording()` on the top-level PySRWrapper.
    Stop recording with `.stop_recording()`. You may then
    use PySR to fit a symbolic regression model to the
    model with `.distill()`.

    Parameters
    ----------
    model : torch.nn.Module
        The model to fit with PySR.
    max_observations : int
        The maximum number of observations to store. Default is `1000`.
    record : bool
        Whether to record observations when `start_recording()` is called.
        Default is `True`.  You may set this to `False` if you wish
        to record observations only about submodules.
    """

    return _create_pysr_wrapper(model, max_observations=max_observations, record=record)
