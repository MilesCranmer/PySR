import inspect

from pysr import PySRRegressor

DEFAULT_PARAMS = inspect.signature(PySRRegressor.__init__).parameters
DEFAULT_NITERATIONS = DEFAULT_PARAMS["niterations"].default
DEFAULT_POPULATIONS = DEFAULT_PARAMS["populations"].default
DEFAULT_NCYCLES = DEFAULT_PARAMS["ncycles_per_iteration"].default
