# Running:

Modify the hyperparameters in `hyperparams.jl` and the dataset in `dataset.jl`
(see below for options). Then, in a new Julia file called
`myfile.jl`, or the interpreter, you can write:

```julia
include("paralleleureqa.jl")
fullRun(10,
    npop=100,
    ncyclesperiteration=1000,
    fractionReplaced=0.1f0,
    verbosity=100,
    topn=10)
```
The first arg is the number of migration periods to run,
with `ncyclesperiteration` determining how many generations
per migration period.  `npop` is the number of population members.
`annealing` determines whether to stay in exploration mode,
or tune it down with each cycle. `fractionReplaced` is
how much of the population is replaced by migrated equations each
step. `topn` is the number of top members of each population
to migrate.


Run it with threading turned on using:
`julia --threads auto -O3 myfile.jl`

## Modification

You can change the binary and unary operators in `hyperparams.jl` here:
```julia
const binops = [plus, mult]
const unaops = [sin, cos, exp];
```
E.g., you can add the function for powers with:
```julia
pow(x::Float32, y::Float32)::Float32 = sign(x)*abs(x)^y
const binops = [plus, mult, pow]
```

You can change the dataset here:
```julia
const X = convert(Array{Float32, 2}, randn(100, 5)*2)
# Here is the function we want to learn (x2^2 + cos(x3) - 5)
const y = convert(Array{Float32, 1}, ((cx,)->cx^2).(X[:, 2]) + cos.(X[:, 3]) .- 5)
```
by either loading in a dataset, or modifying the definition of `y`.
(The `.` are are used for vectorization of a scalar function)

### Hyperparameters

Annealing allows each evolutionary cycle to turn down the exploration
rate over time: at the end (temperature 0), it will only select solutions
better than existing solutions.

The following parameter, parsimony, is how much to punish complex solutions:
```julia
const parsimony = 0.01
```

Finally, the following
determins how much to scale temperature by (T between 0 and 1).
```julia
const alpha = 10.0
```
Larger alpha means more exploration.

One can also adjust the relative probabilities of each operation here:
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
- [ ] Add mutation for constant<->variable
- [ ] Create a Python interface
- [ ] Create a benchmark for accuracy
- [ ] Create struct to pass through all hyperparameters, instead of treating as constants
    - Make sure doesn't affect performance
- [ ] Use NN to generate weights over all probability distribution conditional on error and existing equation, and train on some randomly-generated equations
- [ ] Performance:
    - [ ] Use an enum for functions instead of storing them?
    - Current most expensive operations:
        - [x] deepcopy() before the mutate, to see whether to accept or not.
            - Seems like its necessary right now. But still by far the slowest option.
        - [ ] Calculating the loss function - there is duplicate calculations happening.
        - [ ] Declaration of the weights array every iteration
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
