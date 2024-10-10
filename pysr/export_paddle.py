import collections as co
import functools as ft

import numpy as np  # noqa: F401
import sympy  # type: ignore


def _reduce_add(*args):
    return ft.reduce(lambda a, b: a + b, args)


def _reduce_mul(*args):
    return ft.reduce(lambda a, b: a * b, args)


def _mod(a, b):
    return a % b


def _div(a, b):
    return a / b


paddle_initialized = False
paddle = None
SingleSymPyModule = None


def _initialize_paddle():
    global paddle_initialized
    global paddle
    global SingleSymPyModule

    # Way to lazy load paddle, only if this is called,
    # but still allow this module to be loaded in __init__
    if not paddle_initialized:
        import paddle as _paddle

        paddle = _paddle

        _global_func_lookup = {
            sympy.Mul: _reduce_mul,
            sympy.Add: _reduce_add,
            sympy.div: _div,
            sympy.Abs: paddle.abs,
            sympy.sign: paddle.sign,
            # Note: May raise error for ints.
            sympy.ceiling: paddle.ceil,
            sympy.floor: paddle.floor,
            sympy.log: paddle.log,
            sympy.exp: paddle.exp,
            sympy.sqrt: paddle.sqrt,
            sympy.cos: paddle.cos,
            sympy.acos: paddle.acos,
            sympy.sin: paddle.sin,
            sympy.asin: paddle.asin,
            sympy.tan: paddle.tan,
            sympy.atan: paddle.atan,
            sympy.atan2: paddle.atan2,
            # Note: May give NaN for complex results.
            sympy.cosh: paddle.cosh,
            sympy.acosh: paddle.acosh,
            sympy.sinh: paddle.sinh,
            sympy.asinh: paddle.asinh,
            sympy.tanh: paddle.tanh,
            sympy.atanh: paddle.atanh,
            sympy.Pow: paddle.pow,
            sympy.re: paddle.real,
            sympy.im: paddle.imag,
            sympy.arg: paddle.angle,
            # Note: May raise error for ints and complexes
            sympy.erf: paddle.erf,
            sympy.loggamma: paddle.lgamma,
            sympy.Eq: paddle.equal,
            sympy.Ne: paddle.not_equal,
            sympy.StrictGreaterThan: paddle.greater_than,
            sympy.StrictLessThan: paddle.less_than,
            sympy.LessThan: paddle.less_equal,
            sympy.GreaterThan: paddle.greater_equal,
            sympy.And: paddle.logical_and,
            sympy.Or: paddle.logical_or,
            sympy.Not: paddle.logical_not,
            sympy.Max: paddle.max,
            sympy.Min: paddle.min,
            sympy.Mod: _mod,
            sympy.Heaviside: paddle.heaviside,
            sympy.core.numbers.Half: (lambda: 0.5),
            sympy.core.numbers.One: (lambda: 1.0),
        }

        class _Node(paddle.nn.Layer):
            """Forked from https://github.com/patrick-kidger/sympypaddle"""

            def __init__(self, *, expr, _memodict, _func_lookup, **kwargs):
                super().__init__(**kwargs)

                self._sympy_func = expr.func
                if issubclass(expr.func, sympy.Float):
                    self._value = paddle.create_parameter(
                        shape=[1],
                        dtype="float32",
                        default_initializer=paddle.nn.initializer.Assign(
                            paddle.to_tensor(float(expr))
                        ),
                    )
                    self._paddle_func = lambda: self._value
                    self._args = ()
                elif issubclass(expr.func, sympy.Rational):
                    # This is some fraction fixed in the operator.
                    self._value = float(expr)
                    self._paddle_func = lambda: self._value
                    self._args = ()
                elif issubclass(expr.func, sympy.UnevaluatedExpr):
                    if len(expr.args) != 1 or not issubclass(
                        expr.args[0].func, sympy.Float
                    ):
                        raise ValueError(
                            "UnevaluatedExpr should only be used to wrap floats."
                        )
                    self.register_buffer(
                        "_value", paddle.to_tensor(float(expr.args[0]))
                    )
                    self._paddle_func = lambda: self._value
                    self._args = ()
                elif issubclass(expr.func, sympy.Integer):
                    # Can get here if expr is one of the Integer special cases,
                    # e.g. NegativeOne
                    self._value = int(expr)
                    self._paddle_func = lambda: self._value
                    self._args = ()
                elif issubclass(expr.func, sympy.NumberSymbol):
                    # Can get here from exp(1) or exact pi
                    self._value = float(expr)
                    self._paddle_func = lambda: self._value
                    self._args = ()
                elif issubclass(expr.func, sympy.Symbol):
                    self._name = expr.name
                    self._paddle_func = lambda value: value
                    self._args = ((lambda memodict: memodict[expr.name]),)
                else:
                    try:
                        self._paddle_func = _func_lookup[expr.func]
                    except KeyError:
                        raise KeyError(
                            f"Function {expr.func} was not found in paddle function mappings."
                            "Please add it to extra_paddle_mappings in the format, e.g., "
                            "{sympy.sqrt: paddle.sqrt}."
                        )
                    args = []
                    for arg in expr.args:
                        try:
                            arg_ = _memodict[arg]
                        except KeyError:
                            arg_ = type(self)(
                                expr=arg,
                                _memodict=_memodict,
                                _func_lookup=_func_lookup,
                                **kwargs,
                            )
                            _memodict[arg] = arg_
                        args.append(arg_)
                    self._args = paddle.nn.LayerList(args)

            def extra_repr(self):
                return (
                    f"sympy_func={self._paddle_func.__name__}" + f"{self._sympy_func}"
                )

            def forward(self, memodict):
                args = []
                for arg in self._args:
                    try:
                        arg_ = memodict[arg]
                    except KeyError:
                        arg_ = arg(memodict)
                        memodict[arg] = arg_
                    args.append(arg_)
                return self._paddle_func(*args)

        class _SingleSymPyModule(paddle.nn.Layer):

            def __init__(
                self, expression, symbols_in, selection=None, extra_funcs=None, **kwargs
            ):
                super().__init__(**kwargs)

                if extra_funcs is None:
                    extra_funcs = {}
                _func_lookup = co.ChainMap(_global_func_lookup, extra_funcs)

                _memodict = {}
                self._node = _Node(
                    expr=expression, _memodict=_memodict, _func_lookup=_func_lookup
                )
                self._expression_string = str(expression)
                self._selection = selection
                self.symbols_in = [str(symbol) for symbol in symbols_in]

            def __repr__(self):
                return f"{type(self).__name__}(expression={self._expression_string})"

            def forward(self, X):

                if self._selection is not None:
                    X = X[:, self._selection]
                symbols = {symbol: X[:, i] for i, symbol in enumerate(self.symbols_in)}
                return self._node(symbols)

        SingleSymPyModule = _SingleSymPyModule


def sympy2paddle(expression, symbols_in, selection=None, extra_paddle_mappings=None):
    """Returns a module for a given sympy expression with trainable parameters;

    This function will assume the input to the module is a matrix X, where
        each column corresponds to each symbol you pass in `symbols_in`.
    """
    global SingleSymPyModule

    _initialize_paddle()

    return SingleSymPyModule(
        expression, symbols_in, selection=selection, extra_funcs=extra_paddle_mappings
    )
