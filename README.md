# PySR.jl

[![DOI](https://zenodo.org/badge/295391759.svg)](https://zenodo.org/badge/latestdoi/295391759)
[![PyPI version](https://badge.fury.io/py/pysr.svg)](https://badge.fury.io/py/pysr)
[![Build Status](https://travis-ci.com/MilesCranmer/PySR.svg?branch=master)](https://travis-ci.com/MilesCranmer/PySR)

**Symbolic regression built on Julia, and interfaced by Python.
Uses regularized evolution, simulated annealing, and gradient-free optimization.**

Symbolic regression is a very interpretable machine learning algorithm
for low-dimensional problems: these tools search equation space
to find algebraic relations that approximate a dataset.

One can also
extend these approaches to higher-dimensional
spaces by using a neural network as proxy, as explained in 
https://arxiv.org/abs/2006.11287, where we apply
it to N-body problems. Here, one essentially uses
symbolic regression to convert a neural net
to an analytic equation. Thus, these tools simultaneously present
an explicit and powerful way to interpret deep models.


*Backstory:*

Previously, we have used
[eureqa](https://www.creativemachineslab.com/eureqa.html),
which is a very efficient and user-friendly tool. However,
eureqa is GUI-only, doesn't allow for user-defined
operators, has no distributed capabilities,
and has become proprietary (and recently been merged into an online
service). Thus, the goal
of this package is to have an open-source symbolic regression tool
as efficient as eureqa, while also exposing a configurable
python interface.


# Installation
PySR uses both Julia and Python, so you need to have both installed.

Install Julia - see [downloads](https://julialang.org/downloads/), and
then instructions for [mac](https://julialang.org/downloads/platform/#macos)
and [linux](https://julialang.org/downloads/platform/#linux_and_freebsd).
Then, at the command line,
install the `Optim` and `SpecialFunctions` packages via:

```bash
julia -e 'import Pkg; Pkg.add("Optim"); Pkg.add("SpecialFunctions")'
```

For python, you need to have Python 3, numpy, sympy, and pandas installed.

You can install this package from PyPI with:

```bash
pip install pysr
```

# Quickstart

```python
import numpy as np
from pysr import pysr

# Dataset
X = 2*np.random.randn(100, 5)
y = 2*np.cos(X[:, 3]) + X[:, 0]**2 - 2

# Learn equations
equations = pysr(X, y, niterations=5,
            binary_operators=["plus", "mult"],
            unary_operators=["cos", "exp", "sin"])

...

print(equations)
```

which gives:

```
   Complexity       MSE                                                Equation
0           5  1.947431                          plus(-1.7420927, mult(x0, x0))
1           8  0.486858           plus(-1.8710494, plus(cos(x3), mult(x0, x0)))
2          11  0.000000  plus(plus(mult(x0, x0), cos(x3)), plus(-2.0, cos(x3)))
```

The newest version of PySR also returns three additional columns:

- `score` - a metric akin to Occam's razor; you should use this to help select the "true" equation.
- `sympy_format` - sympy equation.
- `lambda_format` - a lambda function for that equation, that you can pass values through.

### Custom operators

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

We also define `extra_sympy_mappings`,
so that the SymPy code can understand the output equation from Julia,
when constructing a useable function. This step is optional, but
is necessary for the `lambda_format` to work.

One can also edit `operators.jl`. See below for more options.

### Weighted data

Here, we assign weights to each row of data
using inverse uncertainty squared. We also use 10 processes
instead of the usual 4, which creates more populations
(one population per thread).
```python
sigma = ...
weights = 1/sigma**2

equations = pysr.pysr(X, y, weights=weights, procs=10)
```



# Operators

All Base julia operators that take 1 or 2 float32 as input,
and output a float32 as output, are available. A selection
of these and other valid operators are stated below. You can also
define your own in `operators.jl`, and pass the function
name as a string.

Your operator should work with the entire real line (you can use
abs(x) - see `logm`); otherwise
the search code will be slowed down with domain errors.

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

# Full API

What follows is the API reference for running the numpy interface.
You likely don't need to tune the hyperparameters yourself,
but if you would like, you can use `hyperparamopt.py` as an example.
However, you should adjust `procs`, `niterations`,
`binary_operators`, `unary_operators`, and `maxsize`
to your requirements.

The program will output a pandas DataFrame containing the equations,
mean square error, and complexity. It will also dump to a csv
at the end of every iteration,
which is `hall_of_fame.csv` by default. It also prints the
equations to stdout.

```python
pysr(X=None, y=None, weights=None, procs=4, populations=None, niterations=100, ncyclesperiteration=300, binary_operators=["plus", "mult"], unary_operators=["cos", "exp", "sin"], alpha=0.1, annealing=True, fractionReplaced=0.10, fractionReplacedHof=0.10, npop=1000, parsimony=1e-4, migration=True, hofMigration=True, shouldOptimizeConstants=True, topn=10, weightAddNode=1, weightInsertNode=3, weightDeleteNode=3, weightDoNothing=1, weightMutateConstant=10, weightMutateOperator=1, weightRandomize=1, weightSimplify=0.01, perturbationFactor=1.0, nrestarts=3, timeout=None, extra_sympy_mappings={}, equation_file='hall_of_fame.csv', test='simple1', verbosity=1e9, maxsize=20, fast_cycle=False, maxdepth=None, variable_names=[], select_k_features=None, threads=None, julia_optimization=3)
```

Run symbolic regression to fit f(X[i, :]) ~ y[i] for all i.
Note: most default parameters have been tuned over several example
equations, but you should adjust `threads`, `niterations`,
`binary_operators`, `unary_operators` to your requirements.

**Arguments**:

- `X`: np.ndarray or pandas.DataFrame, 2D array. Rows are examples,
columns are features. If pandas DataFrame, the columns are used
for variable names (so make sure they don't contain spaces).
- `y`: np.ndarray, 1D array. Rows are examples.
- `weights`: np.ndarray, 1D array. Each row is how to weight the
mean-square-error loss on weights.
- `procs`: int, Number of processes (=number of populations running).
- `populations`: int, Number of populations running; by default=procs.
- `niterations`: int, Number of iterations of the algorithm to run. The best
equations are printed, and migrate between populations, at the
end of each.
- `ncyclesperiteration`: int, Number of total mutations to run, per 10
samples of the population, per iteration.
- `binary_operators`: list, List of strings giving the binary operators
in Julia's Base, or in `operator.jl`.
- `unary_operators`: list, Same but for operators taking a single `Float32`.
- `alpha`: float, Initial temperature.
- `annealing`: bool, Whether to use annealing. You should (and it is default).
- `fractionReplaced`: float, How much of population to replace with migrating
equations from other populations.
- `fractionReplacedHof`: float, How much of population to replace with migrating
equations from hall of fame.
- `npop`: int, Number of individuals in each population
- `parsimony`: float, Multiplicative factor for how much to punish complexity.
- `migration`: bool, Whether to migrate.
- `hofMigration`: bool, Whether to have the hall of fame migrate.
- `shouldOptimizeConstants`: bool, Whether to numerically optimize
constants (Nelder-Mead/Newton) at the end of each iteration.
- `topn`: int, How many top individuals migrate from each population.
- `nrestarts`: int, Number of times to restart the constant optimizer
- `perturbationFactor`: float, Constants are perturbed by a max
factor of (perturbationFactor\*T + 1). Either multiplied by this
or divided by this.
- `weightAddNode`: float, Relative likelihood for mutation to add a node
- `weightInsertNode`: float, Relative likelihood for mutation to insert a node
- `weightDeleteNode`: float, Relative likelihood for mutation to delete a node
- `weightDoNothing`: float, Relative likelihood for mutation to leave the individual
- `weightMutateConstant`: float, Relative likelihood for mutation to change
the constant slightly in a random direction.
- `weightMutateOperator`: float, Relative likelihood for mutation to swap
an operator.
- `weightRandomize`: float, Relative likelihood for mutation to completely
delete and then randomly generate the equation
- `weightSimplify`: float, Relative likelihood for mutation to simplify
constant parts by evaluation
- `timeout`: float, Time in seconds to timeout search
- `equation_file`: str, Where to save the files (.csv separated by |)
- `test`: str, What test to run, if X,y not passed.
- `maxsize`: int, Max size of an equation.
- `maxdepth`: int, Max depth of an equation. You can use both maxsize and maxdepth.
maxdepth is by default set to = maxsize, which means that it is redundant.
- `fast_cycle`: bool, (experimental) - batch over population subsamples. This
is a slightly different algorithm than regularized evolution, but does cycles
15% faster. May be algorithmically less efficient.
- `variable_names`: list, a list of names for the variables, other
than "x0", "x1", etc.
- `select_k_features`: (None, int), whether to run feature selection in
Python using random forests, before passing to the symbolic regression
code. None means no feature selection; an int means select that many
features.
- `julia_optimization`: int, Optimization level (0, 1, 2, 3)

**Returns**:

pd.DataFrame, Results dataframe, giving complexity, MSE, and equations
(as strings).


# TODO

- [x] Async threading, and have a server of equations. So that threads aren't waiting for others to finish.
- [x] Print out speed of equation evaluation over time. Measure time it takes per cycle
- [x] Add ability to pass an operator as an anonymous function string. E.g., `binary_operators=["g(x, y) = x+y"]`.
- [x] Add error bar capability (thanks Johannes Buchner for suggestion)
- [x] Why don't the constants continually change? It should optimize them every time the equation appears.
    - Restart the optimizer to help with this.
- [x] Add several common unary and binary operators; list these.
- [x] Try other initial conditions for optimizer
- [x] Make scaling of changes to constant a hyperparameter
- [x] Make deletion op join deleted subtree to parent
- [x] Update hall of fame every iteration?
    - Seems to overfit early if we do this.
- [x] Consider adding mutation to pass an operator in through a new binary operator (e.g., exp(x3)->plus(exp(x3), ...))
    - (Added full insertion operator
- [x] Add a node at the top of a tree
- [x] Insert a node at the top of a subtree
- [x] Record very best individual in each population, and return at end.
- [x] Write our own tree copy operation; deepcopy() is the slowest operation by far.
- [x] Hyperparameter tune
- [x] Create a benchmark for accuracy
- [x] Add interface for either defining an operation to learn, or loading in arbitrary dataset.
    - Could just write out the dataset in julia, or load it.
- [x] Create a Python interface
- [x] Explicit constant optimization on hall-of-fame
    - Create method to find and return all constants, from left to right
    - Create method to find and set all constants, in same order
    - Pull up some optimization algorithm and add it. Keep the package small!
- [x] Create a benchmark for speed
- [x] Simplify subtrees with only constants beneath them. Or should I? Maybe randomly simplify sometimes?
- [x] Record hall of fame
- [x] Optionally (with hyperparameter) migrate the hall of fame, rather than current bests
- [x] Test performance of reduced precision integers
    - No effect
- [x] Create struct to pass through all hyperparameters, instead of treating as constants
    - Make sure doesn't affect performance
- [x] Rename package to avoid trademark issues
    - PySR?
- [x] Put on PyPI
- [x] Treat baseline as a solution.
- [x] Print score alongside MSE: \delta \log(MSE)/\delta \log(complexity)
- [x] Calculating the loss function - there is duplicate calculations happening.
- [x] Declaration of the weights array every iteration
- [x] Sympy evaluation
- [x] Threaded recursion
- [x] Test suite
- [x] Performance: - Use an enum for functions instead of storing them?
    - Gets ~40% speedup on small test.
- [x] Use @fastmath
- [x] Try @spawn over each sub-population. Do random sort, compute mutation for each, then replace 10% oldest.
- [x] Control max depth, rather than max number of nodes?
- [x] Allow user to pass names for variables - use these when printing
- [x] Check for domain errors in an equation quickly before actually running the entire array over it. (We do this now recursively - every single equation is checked for nans/infs when being computed.)
- [ ] Sort these todo lists by priority

## Feature ideas

- [ ] Cross-validation
- [ ] read the docs page
- [ ] Sympy printing
- [ ] Better cleanup of zombie processes after <ctl-c>
- [ ] Hierarchical model, so can re-use functional forms. Output of one equation goes into second equation?
- [ ] Call function to read from csv after running, so dont need to run again
- [ ] Add function to plot equations
- [ ] Refresh screen rather than dumping to stdout?
- [ ] Add ability to save state from python
- [ ] Additional degree operators?
- [ ] Multi targets (vector ops). Idea 1: Node struct contains argument for which registers it is applied to. Then, can work with multiple components simultaneously. Though this may be tricky to get right. Idea 2: each op is defined by input/output space. Some operators are flexible, and the spaces should be adjusted automatically. Otherwise, only consider ops that make a tree possible. But will need additional ops here to get it to work. Idea 3: define each equation in 2 parts: one part that is shared between all outputs, and one that is different between all outputs. Maybe this could be an array of nodes corresponding to each output. And those nodes would define their functions.
- [ ] Tree crossover? I.e., can take as input a part of the same equation, so long as it is the same level or below?
- [ ] Consider printing output sorted by score, not by complexity.
- [ ] Dump scores alongside MSE to .csv (and return with Pandas).
- [ ] Create flexible way of providing "simplification recipes." I.e., plus(plus(T, C), C) => plus(T, +(C, C)). The user could pass these.
- [ ] Consider allowing multi-threading turned off, for faster testing (cache issue on travis). Or could simply fix the caching issue there.
- [ ] Consider returning only the equation of interest; rather than all equations.
- [ ] Enable derivative operators. These would differentiate their right argument wrt their left argument, some input variable.

## Algorithmic performance ideas:

- [ ] Idea: use gradient of equation with respect to each operator (perhaps simply add to each operator) to tell which part is the most "sensitive" to changes. Then, perhaps insert/delete/mutate on that part of the tree?
- [ ] Start populations staggered; so that there is more frequent printing (and pops that start a bit later get hall of fame already)?
- [ ] Consider adding mutation for constant<->variable
- [ ] Implement more parts of the original Eureqa algorithms: https://www.creativemachineslab.com/eureqa.html
- [ ] Experiment with freezing parts of model; then we only append/delete at end of tree.
- [ ] Use NN to generate weights over all probability distribution conditional on error and existing equation, and train on some randomly-generated equations
- [ ] For hierarchical idea: after running some number of iterations, do a search for "most common pattern". Then, turn that subtree into its own operator.
- [ ] Calculate feature importances based on features we've already seen, then weight those features up in all random generations.
- [ ] Calculate feature importances of future mutations, by looking at correlation between residual of model, and the features.
    - Store feature importances of future, and periodically update it.
- [ ] Punish depth rather than size, as depth really hurts during optimization.


## Code performance ideas:

- [ ] Try defining a binary tree as an array, rather than a linked list. See https://stackoverflow.com/a/6384714/2689923
- [ ] Add true multi-node processing, with MPI, or just file sharing. Multiple populations per core.
    - Ongoing in cluster branch
- [ ] Performance: try inling things?
- [ ] Try storing things like number nodes in a tree; then can iterate instead of counting

```julia
mutable struct Tree
    degree::Array{Integer, 1}
    val::Array{Float32, 1}
    constant::Array{Bool, 1}
    op::Array{Integer, 1}
    Tree(s::Integer) = new(zeros(Integer, s), zeros(Float32, s), zeros(Bool, s), zeros(Integer, s))
end
```

- Then, we could even work with trees on the GPU, since they are all pre-allocated arrays.
- A population could be a Tree, but with degree 2 on all the degrees. So a slice of population arrays forms a tree.
- How many operations can we do via matrix ops? Mutate node=>easy.
- Can probably batch and do many operations at once across a population.
    - Or, across all populations! Mutate operator: index 2D array and set it to random vector? But the indexing might hurt.
- The big advantage: can evaluate all new mutated trees at once; as massive matrix operation.
    - Can control depth, rather than maxsize. Then just pretend all trees are full and same depth. Then we really don't need to care about depth.

- [ ] Can we cache calculations, or does the compiler do that? E.g., I should only have to run exp(x0) once; after that it should be read from memory.
    - Done on caching branch. Currently am finding that this is quiet slow (presumably because memory allocation is the main issue).
- [ ] Add GPU capability?
     - Not sure if possible, as binary trees are the real bottleneck.
     - Could generate on CPU, evaluate score on GPU?
