"""This module loads export_torch.py when `sympy2torch` is called."""
SingleSymPyModule = None


def _initialize_torch():
    global SingleSymPyModule
    # Way to lazy load torch, only if this is called,
    # but still allow this module to be loaded in __init__
    from .export_torch import SingleSymPyModule as _SingleSymPyModule

    SingleSymPyModule = _SingleSymPyModule


def sympy2torch(expression, symbols_in, selection=None, extra_torch_mappings=None):
    """Returns a module for a given sympy expression with trainable parameters;

    This function will assume the input to the module is a matrix X, where
        each column corresponds to each symbol you pass in `symbols_in`.
    """
    global SingleSymPyModule

    _initialize_torch()

    return SingleSymPyModule(
        expression, symbols_in, selection=selection, extra_funcs=extra_torch_mappings
    )
