from abc import ABC, abstractmethod
from dataclasses import dataclass

from .julia_import import AnyValue, jl


class AbstractLoggerSpec(ABC):
    """Abstract base class for logger specifications."""

    @abstractmethod
    def create_logger(self) -> AnyValue:
        """Create a logger instance."""
        pass


@dataclass
class TensorBoardLoggerSpec(AbstractLoggerSpec):
    """Specification for TensorBoard logger.

    Attributes:
    ----------
    log_dir : str
        Directory where TensorBoard logs will be saved.
    log_interval : int, optional
        Interval (in steps) at which logs are written. Default is 2.
    overwrite : bool, optional
        Whether to overwrite existing logs in the directory. Default is True.
    """

    log_dir: str
    log_interval: int = 2
    overwrite: bool = True

    def create_logger(self) -> AnyValue:
        make_logger = jl.seval(
            """
            function make_logger(log_dir::AbstractString, overwrite::Bool, log_interval::Int)
                base_logger = TensorBoardLogger.TBLogger(
                    log_dir,
                    (overwrite ? (TensorBoardLogger.tb_overwrite,) : ())...
                )
                return SRLogger(; logger=base_logger, log_interval)
            end
        """
        )
        log_dir = str(self.log_dir)
        return make_logger(log_dir, self.overwrite, self.log_interval)
