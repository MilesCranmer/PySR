# PySR.jl

**Symbolic regression built on Julia, and interfaced by Python.
Uses regularized evolution and simulated annealing.**

Backstory: we used the original
[eureqa](https://www.creativemachineslab.com/eureqa.html)
in our [paper](https://arxiv.org/abs/2006.11287) to
convert a graph neural network into
an analytic equation describing dark matter overdensity. However,
eureqa is GUI-only, doesn't allow for user-defined
operators, has no distributed capabilities,
and has become proprietary. Thus, the goal
of this package is to have an open-source symbolic regression tool
as efficient as eureqa, while also exposing a configurable
python interface.

The algorithms here implement regularized evolution, as in
[AutoML-Zero](https://arxiv.org/abs/2003.03384),
but with additional algorithmic changes such as simulated
annealing, and classical optimization of constants.


## Installation

Install [Julia](https://julialang.org/downloads/). Then, at the command line,
install the `Optim` package via: `julia -e 'import Pkg; Pkg.add("Optim")'`.
For python, you need to have Python 3, numpy, and pandas installed.

## Running:

### Quickstart

```python
import numpy as np
from pysr import pysr

# Dataset
X = 2*np.random.randn(100, 5)
y = 2*np.cos(X[:, 3]) + X[:, 0]**2 - 2

# Learn equations
equations = pysr(X, y, niterations=5)

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

### API

What follows is the API reference for running the numpy interface.
You likely don't need to tune the hyperparameters yourself,
but if you would like, you can use `hyperopt.py` as an example.
However, you should adjust `threads`, `niterations`,
`binary_operators`, `unary_operators`, and `maxsize`
to your requirements.

The program will output a pandas DataFrame containing the equations,
mean square error, and complexity. It will also dump to a csv
at the end of every iteration,
which is `hall_of_fame.csv` by default. It also prints the
equations to stdout.

You can add more operators in `operators.jl`, or use default
Julia ones. Make sure all operators are defined for scalar `Float32`.
Then just specify the operator names in your call, as above.
You can also change the dataset learned on by passing in `X` and `y` as
numpy arrays to `pysr(...)`.

```python
pysr(X=None, y=None, threads=4, niterations=20,
   ncyclesperiteration=int(default_ncyclesperiteration),
   binary_operators=["plus", "mult"], unary_operators=["cos", "exp", "sin"],
   alpha=default_alpha, annealing=True, fractionReplaced=default_fractionReplaced,
   fractionReplacedHof=default_fractionReplacedHof, npop=int(default_npop),
   parsimony=default_parsimony, migration=True, hofMigration=True
   shouldOptimizeConstants=True, topn=int(default_topn),
   weightAddNode=default_weightAddNode, weightDeleteNode=default_weightDeleteNode,
   weightDoNothing=default_weightDoNothing,
   weightMutateConstant=default_weightMutateConstant,
   weightMutateOperator=default_weightMutateOperator,
   weightRandomize=default_weightRandomize, weightSimplify=default_weightSimplify,
   timeout=None, equation_file='hall_of_fame.csv', test='simple1', maxsize=20)
```

Run symbolic regression to fit f(X[i, :]) ~ y[i] for all i.

**Arguments**:

- `X`: np.ndarray, 2D array. Rows are examples, columns are features.
- `y`: np.ndarray, 1D array. Rows are examples.
- `threads`: int, Number of threads (=number of populations running).
You can have more threads than cores - it actually makes it more
efficient.
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
- `weightAddNode`: float, Relative likelihood for mutation to add a node
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

**Returns**:

pd.DataFrame, Results dataframe, giving complexity, MSE, and equations
(as strings).


# TODO

- [ ] Rename package to avoid trademark issues
    - PySR?
- [ ] Calculate feature importances of future mutations, by looking at correlation between residual of model, and the features.
    - Store feature importances of future, and periodically update it.
- [ ] Implement more parts of the original Eureqa algorithms: https://www.creativemachineslab.com/eureqa.html
- [ ] Sympy printing
- [ ] Consider adding mutation for constant<->variable
- [ ] Hierarchical model, so can re-use functional forms. Output of one equation goes into second equation?
- [ ] Use NN to generate weights over all probability distribution conditional on error and existing equation, and train on some randomly-generated equations
- [ ] Performance:
    - [ ] Use an enum for functions instead of storing them?
    - Current most expensive operations:
        - [ ] Calculating the loss function - there is duplicate calculations happening.
        - [x] Declaration of the weights array every iteration
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
