# Operators

## Pre-defined

All Base julia operators that take 1 or 2 float32 as input,
and output a float32 as output, are available. A selection
of these and other valid operators are stated below.

**Binary**

`plus`, `mult`, `pow`, `div`, `greater`, `mod`, `beta`, `logical_or`,
`logical_and`

**Unary**

`neg`,
`exp`,
`abs`,
`logm` (=log(abs(x) + 1e-8)),
`logm10` (=log10(abs(x) + 1e-8)),
`logm2` (=log2(abs(x) + 1e-8)),
`sqrtm` (=sqrt(abs(x)))
`log1p`,
`sin`,
`cos`,
`tan`,
`sinh`,
`cosh`,
`tanh`,
`asin`,
`acos`,
`atan`,
`asinh`,
`acosh`,
`atanh`,
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
    pysr(
        ...,
        unary_operators=["myfunction(x) = x^2"],
        binary_operators=["myotherfunction(x, y) = x^2*y"]
    )
```


You can also define your own in `julia/operators.jl`,
and pass the function name as a string. This is suitable
for more complex functions. Make sure that it works with
`Float32` as a datatype. That means you need to write `1.5f3`
instead of `1.5e3`, if you write any constant numbers.

Your operator should work with the entire real line (you can use
abs(x) - see `logm`); otherwise
the search code will be slowed down with domain errors.


