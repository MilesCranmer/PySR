# Eureqa.jl

Symbolic regression built on Julia, and interfaced by Python.
Uses regularized evolution and simulated annealing.

## Running:

You can either call the program by calling the `eureqa` function from `eureqa.py`,
or execute the program from the command line with, for example:
```bash
python eureqa.py --threads 8 --binary-operators plus mult pow --npop 200
```

Here is the full list of arguments:
```
usage: eureqa.py [-h] [--threads THREADS] [--parsimony PARSIMONY]
                 [--alpha ALPHA] [--maxsize MAXSIZE]
                 [--niterations NITERATIONS] [--npop NPOP]
                 [--ncyclesperiteration NCYCLESPERITERATION] [--topn TOPN]
                 [--fractionReplacedHof FRACTIONREPLACEDHOF]
                 [--fractionReplaced FRACTIONREPLACED] [--migration MIGRATION]
                 [--hofMigration HOFMIGRATION]
                 [--shouldOptimizeConstants SHOULDOPTIMIZECONSTANTS]
                 [--annealing ANNEALING] [--equation_file EQUATION_FILE]
                 [--test TEST]
                 [--binary-operators BINARY_OPERATORS [BINARY_OPERATORS ...]]
                 [--unary-operators UNARY_OPERATORS]

optional arguments:
  -h, --help            show this help message and exit
  --threads THREADS     Number of threads (default: 4)
  --parsimony PARSIMONY
                        How much to punish complexity (default: 0.001)
  --alpha ALPHA         Scaling of temperature (default: 10)
  --maxsize MAXSIZE     Max size of equation (default: 20)
  --niterations NITERATIONS
                        Number of total migration periods (default: 20)
  --npop NPOP           Number of members per population (default: 100)
  --ncyclesperiteration NCYCLESPERITERATION
                        Number of evolutionary cycles per migration (default:
                        5000)
  --topn TOPN           How many best species to distribute from each
                        population (default: 10)
  --fractionReplacedHof FRACTIONREPLACEDHOF
                        Fraction of population to replace with hall of fame
                        (default: 0.1)
  --fractionReplaced FRACTIONREPLACED
                        Fraction of population to replace with best from other
                        populations (default: 0.1)
  --migration MIGRATION
                        Whether to migrate (default: True)
  --hofMigration HOFMIGRATION
                        Whether to have hall of fame migration (default: True)
  --shouldOptimizeConstants SHOULDOPTIMIZECONSTANTS
                        Whether to use classical optimization on constants
                        before every migration (doesn't impact performance
                        that much) (default: True)
  --annealing ANNEALING
                        Whether to use simulated annealing (default: True)
  --equation_file EQUATION_FILE
                        File to dump best equations to (default:
                        hall_of_fame.csv)
  --test TEST           Which test to run (default: simple1)
  --binary-operators BINARY_OPERATORS [BINARY_OPERATORS ...]
                        Binary operators. Make sure they are defined in
                        operators.jl (default: ['plus', 'mul'])
  --unary-operators UNARY_OPERATORS
                        Unary operators. Make sure they are defined in
                        operators.jl (default: ['exp', 'sin', 'cos'])
```




## Modification

You can add more operators in `operators.jl`, or use default
Julia ones. Make sure all operators are defined for scalar `Float32`.
Then just specify the operator names in your call, as above.
You can also change the dataset learned on by passing in `X` and `y` as
numpy arrays to `eureqa(...)`.

One can also adjust the relative probabilities of each operation here,
inside `eureqa.jl`:
```julia
weights = [8, 1, 1, 1, 0.1, 0.5, 2]
```
for:

1. Perturb constant
2. Mutate operator
3. Append a node
4. Delete a subtree
5. Simplify equation
6. Randomize completely
7. Do nothing


# TODO

- [ ] Hyperparameter tune
- [ ] Add interface for either defining an operation to learn, or loading in arbitrary dataset.
    - Could just write out the dataset in julia, or load it.
- [ ] Add mutation for constant<->variable
- [ ] Create a benchmark for accuracy
- [ ] Use NN to generate weights over all probability distribution conditional on error and existing equation, and train on some randomly-generated equations
- [ ] Performance:
    - [ ] Use an enum for functions instead of storing them?
    - Current most expensive operations:
        - [x] deepcopy() before the mutate, to see whether to accept or not.
            - Seems like its necessary right now. But still by far the slowest option.
        - [ ] Calculating the loss function - there is duplicate calculations happening.
        - [ ] Declaration of the weights array every iteration
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
