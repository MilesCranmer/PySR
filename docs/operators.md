# Operators

## Pre-defined

All Base julia operators that take 1 or 2 scalars as input,
and output a scalar as output, are available. A selection
of these and other valid operators are stated below.

**Binary**

`+`, `-`, `*`, `/`, `^`, `greater`, `mod`, `logical_or`,
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
`Float32` as a datatype (for default precision, or `Float64` if you set `precision=64`). That means you need to write `1.5f3`
instead of `1.5e3`, if you write any constant numbers, or simply convert a result to `Float64(...)`.

PySR expects that operators not throw an error for any input value over the entire real line from `-3.4e38` to `+3.4e38`.
Thus, for "invalid" inputs, such as negative numbers to a `sqrt` function, you may simply return a `NaN` of the same type as the input. For example,

```julia
my_sqrt(x) = x >= 0 ? sqrt(x) : convert(typeof(x), NaN)
```

would be a valid operator. The genetic algorithm
will preferentially selection expressions which avoid
any invalid values over the training dataset.
