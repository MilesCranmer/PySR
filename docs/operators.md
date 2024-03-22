# Operators

## Pre-defined

First, note that pretty much any valid Julia function which
takes one or two scalars as input, and returns on scalar as output,
is likely to be a valid operator[^1].
A selection of these and other valid operators are stated below.

**Binary**

- `+`
- `-`
- `*`
- `/`
- `^`
- `max`
- `min`
- `mod`
- `cond`
    - Equal to `(x, y) -> x > 0 ? y : 0`
- `greater`
    - Equal to `(x, y) -> x > y ? 1 : 0`
- `logical_or`
    - Equal to `(x, y) -> (x > 0 || y > 0) ? 1 : 0`
- `logical_and`
    - Equal to `(x, y) -> (x > 0 && y > 0) ? 1 : 0`

**Unary**

- `neg`
- `square`
- `cube`
- `exp`
- `abs`
- `log`
- `log10`
- `log2`
- `log1p`
- `sqrt`
- `sin`
- `cos`
- `tan`
- `sinh`
- `cosh`
- `tanh`
- `atan`
- `asinh`
- `acosh`
- `atanh_clip`
    - Equal to `atanh(mod(x + 1, 2) - 1)`
- `erf`
- `erfc`
- `gamma`
- `relu`
- `round`
- `floor`
- `ceil`
- `sign`

## Custom

Instead of passing a predefined operator as a string,
you can just define a custom function as Julia code. For example:

```python
    PySRRegressor(
        ...,
        unary_operators=["myfunction(x) = x^2"],
        binary_operators=["myotherfunction(x, y) = x^2*y"],
        extra_sympy_mappings={
            "myfunction": lambda x: x**2,
            "myotherfunction": lambda x, y: x**2 * y,
        },
    )
```


Make sure that it works with
`Float32` as a datatype (for default precision, or `Float64` if you set `precision=64`). That means you need to write `1.5f3`
instead of `1.5e3`, if you write any constant numbers, or simply convert a result to `Float64(...)`.

PySR expects that operators not throw an error for any input value over the entire real line from `-3.4e38` to `+3.4e38`.
Thus, for invalid inputs, such as negative numbers to a `sqrt` function, you may simply return a `NaN` of the same type as the input. For example,

```julia
my_sqrt(x) = x >= 0 ? sqrt(x) : convert(typeof(x), NaN)
```

would be a valid operator. The genetic algorithm
will preferentially selection expressions which avoid
any invalid values over the training dataset.


<!-- Footnote for 1: -->
<!-- (Will say "However, you may need to define a `extra_sympy_mapping`":) -->

[^1]: However, you will need to define a sympy equivalent in `extra_sympy_mapping` if you want to use a function not in the above list.
