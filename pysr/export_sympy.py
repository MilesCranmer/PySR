"""Define utilities to export to sympy"""
from typing import Callable, Dict, List, Optional

import sympy
from sympy import sympify

sympy_mappings = {
    "div": lambda x, y: x / y,
    "mult": lambda x, y: x * y,
    "sqrt": lambda x: sympy.sqrt(x),
    "sqrt_abs": lambda x: sympy.sqrt(abs(x)),
    "square": lambda x: x**2,
    "cube": lambda x: x**3,
    "plus": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "neg": lambda x: -x,
    "pow": lambda x, y: x**y,
    "pow_abs": lambda x, y: abs(x) ** y,
    "cos": sympy.cos,
    "sin": sympy.sin,
    "tan": sympy.tan,
    "cosh": sympy.cosh,
    "sinh": sympy.sinh,
    "tanh": sympy.tanh,
    "exp": sympy.exp,
    "acos": sympy.acos,
    "asin": sympy.asin,
    "atan": sympy.atan,
    "acosh": lambda x: sympy.acosh(x),
    "acosh_abs": lambda x: sympy.acosh(abs(x) + 1),
    "asinh": sympy.asinh,
    "atanh": lambda x: sympy.atanh(sympy.Mod(x + 1, 2) - 1),
    "atanh_clip": lambda x: sympy.atanh(sympy.Mod(x + 1, 2) - 1),
    "abs": abs,
    "mod": sympy.Mod,
    "erf": sympy.erf,
    "erfc": sympy.erfc,
    "log": lambda x: sympy.log(x),
    "log10": lambda x: sympy.log(x, 10),
    "log2": lambda x: sympy.log(x, 2),
    "log1p": lambda x: sympy.log(x + 1),
    "log_abs": lambda x: sympy.log(abs(x)),
    "log10_abs": lambda x: sympy.log(abs(x), 10),
    "log2_abs": lambda x: sympy.log(abs(x), 2),
    "log1p_abs": lambda x: sympy.log(abs(x) + 1),
    "floor": sympy.floor,
    "ceil": sympy.ceiling,
    "sign": sympy.sign,
    "gamma": sympy.gamma,
    "max": lambda x, y: sympy.Piecewise((y, x < y), (x, True)),
    "min": lambda x, y: sympy.Piecewise((x, x < y), (y, True)),
    "round": lambda x: sympy.ceiling(x - 0.5),
    "cond": lambda x, y: sympy.Heaviside(x, H0=0) * y,
}


def create_sympy_symbols(
    feature_names_in: List[str],
) -> List[sympy.Symbol]:
    return [sympy.Symbol(variable) for variable in feature_names_in]


def pysr2sympy(
    equation: str, *, extra_sympy_mappings: Optional[Dict[str, Callable]] = None
):
    local_sympy_mappings = {
        **(extra_sympy_mappings if extra_sympy_mappings else {}),
        **sympy_mappings,
    }

    return sympify(equation, locals=local_sympy_mappings)


def assert_valid_sympy_symbol(var_name: str) -> None:
    if var_name in sympy_mappings or var_name in sympy.__dict__.keys():
        raise ValueError(f"Variable name {var_name} is already a function name.")
