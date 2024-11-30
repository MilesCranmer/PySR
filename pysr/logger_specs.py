from abc import ABC, abstractmethod
from dataclasses import dataclass

from .julia_import import AnyValue, jl


class AbstractLoggerSpec(ABC):
    """Abstract base class for logger specifications."""

    @abstractmethod
    def create_logger(self) -> AnyValue:
        """Create a logger instance."""
        pass

    def close(self, logger: AnyValue) -> None:
        """Close the logger."""
        jl.close(logger)


@dataclass
class TensorBoardLoggerSpec(AbstractLoggerSpec):
    """Specification for TensorBoard logger.

    Attributes:
    ----------
    log_dir : str
        Directory where TensorBoard logs will be saved. If `overwrite` is `False`,
        new logs will be saved to `{log_dir}_1`, and so on. Default is `"logs/run"`.
    log_interval : int, optional
        Interval (in steps) at which logs are written. Default is 2.
    overwrite : bool, optional
        Whether to overwrite existing logs in the directory. Default is False.
    """

    log_dir: str = "logs/run"
    log_interval: int = 2
    overwrite: bool = False

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
