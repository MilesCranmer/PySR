from __future__ import annotations

from dataclasses import dataclass

from .julia_import import AnyValue, SymbolicRegression


@dataclass
class BacksolveOptions:
    """Options for the experimental backsolve mutation.

    Parameters
    ----------
    max_library_size : int
        Maximum number of candidate library terms. Default is `500`.
    lambda_ : float
        STLSQ sparsity threshold. Default is `0.01`.
    max_iter : int
        Maximum number of STLSQ iterations. Default is `10`.
    """

    max_library_size: int = 500
    lambda_: float = 0.01
    max_iter: int = 10

    def julia_options(self) -> AnyValue:
        """Create the corresponding `SymbolicRegression.BacksolveOptions`."""
        return SymbolicRegression.BacksolveOptions(
            max_library_size=int(self.max_library_size),
            max_iter=int(self.max_iter),
            **{"lambda": float(self.lambda_)},
        )
