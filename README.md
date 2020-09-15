# Running:

You can run the performance benchmark with `./benchmark.sh`.

Modify the search code in `paralleleureqa.jl` and `eureqa.jl` to your liking
(see below for options). Then, in a new Julia file called
`myfile.jl`, you can write:

```julia
include("paralleleureqa.jl")
fullRun(10,
    npop=100,
    annealing=true,
    ncyclesperiteration=1000,
    fractionReplaced=0.1f0,
    verbosity=100)
```
The first arg is the number of migration periods to run,
with `ncyclesperiteration` determining how many generations
per migration period.  `npop` is the number of population members.
`annealing` determines whether to stay in exploration mode,
or tune it down with each cycle. `fractionReplaced` is
how much of the population is replaced by migrated equations each
step.


Run it with threading turned on using:
`julia --threads auto -O3 myfile.jl`

## Modification

You can change the binary and unary operators in `eureqa.jl` here:
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
# Here is the function we want to learn (x2^2 + cos(x3))
const y = convert(Array{Float32, 1}, ((cx,)->cx^2).(X[:, 2]) + cos.(X[:, 3]))
```
by either loading in a dataset, or modifying the definition of `y`.

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

One can also adjust the relative probabilities of each mutation here:
```julia
weights = [8, 1, 1, 1, 2]
```
(for: 1. perturb constant, 2. mutate operator,
3. append a node, 4. delete a subtree, 5. do nothing).


# TODO

- [ ] Create a Python interface
- [x] Create a benchmark for speed
- [ ] Create a benchmark for accuracy
- [ ] Record hall of fame
- [ ] Optionally (with hyperparameter) migrate the hall of fame, rather than current bests
- [x] Test performance of reduced precision integers
    - No effect
- [ ] Create struct to pass through all hyperparameters, instead of treating as constants
    - Make sure doesn't affect performance
- [ ] Hyperparameter tune
- [ ] Use NN to generate weights over all probability distribution, and train on some randomly-generated equations
- [ ] Performance:
    - Use an enum for functions instead of storing them?

