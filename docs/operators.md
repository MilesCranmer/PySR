# Operators

## Pre-defined

All Base julia operators that take 1 or 2 float32 as input,
and output a float32 as output, are available. A selection
of these and other valid operators are stated below.

**Binary**

`plus`, `sub`, `mult`, `pow`, `div`, `greater`, `mod`, `logical_or`,
`logical_and`

**Unary**

`neg`,
`square`,
`cube`,
`exp`,
`abs`,
`log`,
`log10`,
`log2`,
`log1p`,
`sqrt`,
`sin`,
`cos`,
`tan`,
`sinh`,
`cosh`,
`tanh`,
`atan`,
`asinh`,
`acosh`,
`atanh_clip` (=atanh((x+1)%2 - 1)),
`erf`,
`erfc`,
`gamma`,
`relu`,
`round`,
`floor`,
`ceil`,
`round`,
`sign`.

## Custom

Instead of passing a predefined operator as a string,
you can define with by passing it to the `pysr` function, with, e.g.,

```python
    PySRRegressor(
        ...,
        unary_operators=["myfunction(x) = x^2"],
        binary_operators=["myotherfunction(x, y) = x^2*y"]
    )
```


Make sure that it works with
`Float32` as a datatype. That means you need to write `1.5f3`
instead of `1.5e3`, if you write any constant numbers.

Your operator should work with the entire real line (you can use
abs(x) for operators requiring positive input - see `log_abs`); otherwise
the search code will experience domain errors.


