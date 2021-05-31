import collections as co
import sympy

torch_initialized = False
torch = None
sympytorch = None
PySRTorchModule = None

def _initialize_torch():
    global torch_initialized
    global torch
    global sympytorch
    global PySRTorchModule

    # Way to lazy load torch and sympytorch, only if this is called,
    # but still allow this module to be loaded in __init__
    if not torch_initialized:
        try:
            import torch
            import sympytorch
        except ImportError:
            raise ImportError("You need to pip install `torch` and `sympytorch` before exporting to pytorch.")
        torch_initialized = True


        class PySRTorchModule(torch.nn.Module):
            def __init__(self, *, expression, symbols_in,
                         selection=None, extra_funcs=None, **kwargs):
                super().__init__(**kwargs)
                self._module = sympytorch.SymPyModule(
                        expressions=[expression],
                        extra_funcs=extra_funcs)
                self._selection = selection
                self._symbols = symbols_in
                
            def __repr__(self):
                return f"{type(self).__name__}(expression={self._expression_string})"

            def forward(self, X):
                if self._selection is not None:
                    X = X[:, self._selection]
                symbols = {str(symbol): X[:, i]
                           for i, symbol in enumerate(self._symbols)}
                return self._module(**symbols)[..., 0]


def sympy2torch(expression, symbols_in,
                selection=None, extra_torch_mappings=None):
    """Returns a module for a given sympy expression with trainable parameters;

    This function will assume the input to the module is a matrix X, where
        each column corresponds to each symbol you pass in `symbols_in`.
    """
    global PySRTorchModule

    _initialize_torch()

    return PySRTorchModule(expression=expression,
                           symbols_in=symbols_in,
                           selection=selection,
                           extra_funcs=extra_torch_mappings)
