# Features and Options

You likely don't need to tune the hyperparameters yourself,
but if you would like, you can use `hyperparamopt.py` as an example.

Some configurable features and options in `PySR` which you
may find useful include:
- `binary_operators`, `unary_operators`
- `niterations`
- `ncyclesperiteration`
- `procs`
- `populations`
- `weights`
- `maxsize`, `maxdepth`
- `batching`, `batchSize`
- `variable_names` (or pandas input)
- Constraining operator complexity
- LaTeX, SymPy, and callable equation output
- `loss`

These are described below

The program will output a pandas DataFrame containing the equations,
mean square error, and complexity. It will also dump to a csv
at the end of every iteration,
which is `hall_of_fame_{date_time}.csv` by default. It also prints the
equations to stdout.

## Operators

A list of operators can be found on the operators page.
One can define custom operators in Julia by passing a string:
```python
equations = pysr.pysr(X, y, niterations=100,
    binary_operators=["mult", "plus", "special(x, y) = x^2 + y"],
    extra_sympy_mappings={'special': lambda x, y: x**2 + y},
    unary_operators=["cos"])
```

Now, the symbolic regression code can search using this `special` function
that squares its left argument and adds it to its right. Make sure
all passed functions are valid Julia code, and take one (unary)
or two (binary) float32 scalars as input, and output a float32. This means if you
write any real constants in your operator, like `2.5`, you have to write them
instead as `2.5f0`, which defines it as `Float32`.
Operators are automatically vectorized.

One should also define `extra_sympy_mappings`,
so that the SymPy code can understand the output equation from Julia,
when constructing a useable function. This step is optional, but
is necessary for the `lambda_format` to work.

One can also edit `operators.jl`.

## Iterations

This is the total number of generations that `pysr` will run for.
I usually set this to a large number, and exit when I am satisfied
with the equations.

## Cycles per iteration

Each cycle considers every 10-equation subsample (re-sampled for each individual 10,
unless `fast_cycle` is set in which case the subsamples are separate groups of equations)
a single time, producing one mutated equation for each.
The parameter `ncyclesperiteration` defines how many times this
occurs before the equations are compared to the hall of fame,
and new equations are migrated from the hall of fame, or from other populations.
It also controls how slowly annealing occurs. You may find that increasing
`ncyclesperiteration` results in a higher cycles-per-second, as the head
worker needs to reduce and distribute new equations less often, and also increases
diversity. But at the same
time, a smaller number it might be that migrating equations from the hall of fame helps
each population stay closer to the best current equations.

## Processors

One can adjust the number of workers used by Julia with the
`procs` option. You should set this equal to the number of cores
you want `pysr` to use. This will also run `procs` number of
populations simultaneously by default.

## Populations

By default, `populations=procs`, but you can set a different
number of populations with this option. More populations may increase
the diversity of equations discovered, though will take longer to train.
However, it may be more efficient to have `populations>procs`,
as there are multiple populations running
on each core.

## Weighted data

Here, we assign weights to each row of data
using inverse uncertainty squared. We also use 10 processes
instead of the usual 4, which creates more populations
(one population per thread).
```python
sigma = ...
weights = 1/sigma**2

equations = pysr.pysr(X, y, weights=weights, procs=10)
```

## Max size

`maxsize` controls the maximum size of equation (number of operators,
constants, variables). `maxdepth` is by default not used, but can be set
to control the maximum depth of an equation. These will make processing
faster, as longer equations take longer to test.

One can warm up the maxsize from a small number to encourage
PySR to start simple, by using the `warmupMaxsize` argument.
This specifies that maxsize increases every `warmupMaxsize`.


## Batching
One can turn on mini-batching, with the `batching` flag,
and control the batch size with `batchSize`. This will make
evolution faster for large datasets. Equations are still evaluated
on the entire dataset at the end of each iteration to compare to the hall
of fame, but only on a random subset during mutations and annealing.

## Variable Names

You can pass a list of strings naming each column of `X` with
`variable_names`. Alternatively, you can pass `X` as a pandas dataframe
and the columns will be used as variable names. Make sure only
alphabetical characters and `_` are used in these names.

## Constraining operator complexity

One can limit the complexity of specific operators with the `constraints` parameter.
There is a "maxsize" parameter to PySR, but there is also an operator-level
"constraints" parameter. One supplies a dict, like so:

```python
constraints={'pow': (-1, 1), 'mult': (3, 3), 'cos': 5}
```

What this says is that: a power law x^y can have an expression of arbitrary (-1) complexity in the x, but only complexity 1 (e.g., a constant or variable) in the y. So (x0 + 3)^5.5 is allowed, but 5.5^(x0 + 3) is not.
I find this helps a lot for getting more interpretable equations.
The other terms say that each multiplication can only have sub-expressions
of up to complexity 3 (e.g., 5.0 + x2) in each side, and cosine can only operate on
expressions of complexity 5 (e.g., 5.0 + x2 exp(x3)).

## LaTeX, SymPy, callables

The `pysr` command will return a pandas dataframe. The `sympy_format`
column gives sympy equations, and the `lambda_format` gives callable
functions. These use the variable names you have provided.

There are also some helper functions for doing this quickly.
You can call `get_hof()` (or pass an equation file explicitly to this)
to get this pandas dataframe.

You can call the functions `best()` to get the sympy format
for the best equation, using the `score` column to sort equations.
`best_latex()` returns the LaTeX form of this, and `best_callable()`
returns a callable function.

## `loss`

The default loss is mean-square error, and weighted mean-square error.
One can pass an arbitrary Julia string to define a custom loss, using,
e.g., `loss="myloss(x, y) = abs(x - y)^1.5"`. For more details,
see the
[Losses](https://milescranmer.github.io/SymbolicRegression.jl/dev/losses/)
page for SymbolicRegression.jl.

Here are some additional examples:

abs(x-y) loss
```python
pysr(..., loss="f(x, y) = abs(x - y)^1.5")
```
Note that the function name doesn't matter:
```python
pysr(..., loss="loss(x, y) = abs(x * y)")
```
With weights:
```python
pysr(..., weights=weights, loss="myloss(x, y, w) = w * abs(x - y)") 
```
Weights can be used in arbitrary ways:
```python
pysr(..., weights=weights, loss="myloss(x, y, w) = abs(x - y)^2/w^2")
```
Built-in loss (faster) (see [losses](https://astroautomata.com/SymbolicRegression.jl/dev/losses/)).
This one computes the L3 norm:
```python
pysr(..., loss="LPDistLoss{3}()")
```
Can also uses these losses for weighted (weighted-average):
```python
pysr(..., weights=weights, loss="LPDistLoss{3}()")
```
