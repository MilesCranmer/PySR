"""Export expressions to JAX."""
import sympy
import jax
from jax import numpy as jnp
from jax.scipy import special as jsp

# Special since need to reduce arguments.
MUL = 0
ADD = 1

JNP_FUNC_LOOKUP = {
    sympy.Mul: MUL,
    sympy.Add: ADD,
    sympy.div: "jnp.div",
    sympy.Abs: "jnp.abs",
    sympy.sign: "jnp.sign",
    # Note: May raise error for ints.
    sympy.ceiling: "jnp.ceil",
    sympy.floor: "jnp.floor",
    sympy.log: "jnp.log",
    sympy.exp: "jnp.exp",
    sympy.sqrt: "jnp.sqrt",
    sympy.cos: "jnp.cos",
    sympy.acos: "jnp.acos",
    sympy.sin: "jnp.sin",
    sympy.asin: "jnp.asin",
    sympy.tan: "jnp.tan",
    sympy.atan: "jnp.atan",
    sympy.atan2: "jnp.atan2",
    # Note: Also may give NaN for complex results.
    sympy.cosh: "jnp.cosh",
    sympy.acosh: "jnp.acosh",
    sympy.sinh: "jnp.sinh",
    sympy.asinh: "jnp.asinh",
    sympy.tanh: "jnp.tanh",
    sympy.atanh: "jnp.atanh",
    sympy.Pow: "jnp.power",
    sympy.re: "jnp.real",
    sympy.im: "jnp.imag",
    sympy.arg: "jnp.angle",
    # Note: May raise error for ints and complexes
    sympy.erf: "jsp.erf",
    sympy.erfc: "jsp.erfc",
    sympy.LessThan: "jnp.less",
    sympy.GreaterThan: "jnp.greater",
    sympy.And: "jnp.logical_and",
    sympy.Or: "jnp.logical_or",
    sympy.Not: "jnp.logical_not",
    sympy.Max: "jnp.max",
    sympy.Min: "jnp.min",
    sympy.Mod: "jnp.mod",
    sympy.Heaviside: "jnp.heaviside",
    sympy.core.numbers.Half: "(lambda: 0.5)",
    sympy.core.numbers.One: "(lambda: 1.0)",
}

LEAF_TYPES = (sympy.Float, sympy.Rational, sympy.Integer, sympy.Symbol)


def _is_leaf(expr):
    return any(issubclass(expr.func, leaf_type) for leaf_type in LEAF_TYPES)


def _sympy2jaxtext(expr, parameters, symbols_in, extra_jax_mappings=None):
    if _is_leaf(expr):
        if issubclass(expr.func, sympy.Float):
            parameters.append(float(expr))
            out = f"parameters[{len(parameters) - 1}]"
        elif issubclass(expr.func, sympy.Rational):
            out = f"{float(expr)}"
        elif issubclass(expr.func, sympy.Integer):
            out = f"{int(expr)}"
        elif issubclass(expr.func, sympy.Symbol):
            out = f"X[:, {[i for i in range(len(symbols_in)) if symbols_in[i] == expr][0]}]"
        else:
            raise NotImplementedError
        return out

    if extra_jax_mappings is None:
        extra_jax_mappings = {}
    try:
        _func = {**JNP_FUNC_LOOKUP, **extra_jax_mappings}[expr.func]
    except KeyError:
        raise KeyError(
            f"Function {expr.func} was not found in JAX function mappings."
            "Please add it to extra_jax_mappings in the format, e.g., "
            "{sympy.sqrt: 'jnp.sqrt'}."
        )
    args = [
        _sympy2jaxtext(
            arg, parameters, symbols_in, extra_jax_mappings=extra_jax_mappings
        )
        for arg in expr.args
    ]
    if _func == MUL:
        return " * ".join(["(" + arg + ")" for arg in args])
    if _func == ADD:
        return " + ".join(["(" + arg + ")" for arg in args])
    return f'{_func}({", ".join(args)})'


def _sympy2jax(expression, symbols_in, selection=None, extra_jax_mappings=None):
    parameters = []
    functional_form_text = _sympy2jaxtext(
        expression, parameters, symbols_in, extra_jax_mappings
    )
    hash_string = "A_" + str(abs(hash(str(expression) + str(symbols_in))))
    text = f"def {hash_string}(X, parameters):\n"
    if selection is not None:
        # Impose the feature selection:
        text += f"    X = X[:, {list(selection)}]\n"
    text += "    return "
    text += functional_form_text
    ldict = {}
    exec(text, globals(), ldict)
    return ldict[hash_string], jnp.array(parameters)
