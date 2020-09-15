# Running:

For now, just modify the script in `paralleleureqa.jl`
to your liking and run:

`julia --threads auto -O3 paralleleureqa.jl`

## Modification

You can change the binary and unary operators in `eureqa.jl` here:
```
const binops = [plus, mult]
const unaops = [sin, cos, exp];
```
E.g., you can add the function for powers with:
```
pow(x::Float32, y::Float32)::Float32 = sign(x)*abs(x)^y
const binops = [plus, mult, pow]
```

You can change the dataset here:
```
const X = convert(Array{Float32, 2}, randn(100, 5)*2)
# Here is the function we want to learn (x2^2 + cos(x3))
const y = convert(Array{Float32, 1}, ((cx,)->cx^2).(X[:, 2]) + cos.(X[:, 3]))
```
by either loading in a dataset, or modifying the definition of `y`.

### Hyperparameters

Turn on annealing by setting the following in `paralleleureqa.jl`:

`const annealing = true`

Annealing allows each evolutionary cycle to turn down the exploration
rate over time: at the end (temperature 0), it will only select solutions
better than existing solutions.

The following parameter, parsimony, is how much to punish complex solutions:
`
const parsimony = 0.01
`

Finally, the following
determins how much to scale temperature by (T between 0 and 1).
`
const alpha = 10.0
`
Larger alpha means more exploration.

One can also adjust the relative probabilities of each mutation here:
```
weights = [8, 1, 1, 1, 2]
```
(for: 1. perturb constant, 2. mutate operator,
3. append a node, 4. delete a subtree, 5. do nothing).


# TODO

- Create a Python interface
- Create a benchmark for speed
- Create a benchmark for accuracy
- Record hall of fame
- Optionally (with hyperparameter) migrate the hall of fame, rather than current bests
- Create struct to pass through all hyperparameters, instead of treating as constants
    - Make sure doesn't affect performance
- Hyperparameter tune
- Use NN to generate weights over all probability distribution, and train on some randomly-generated equations
- Performance:
    - Use an enum for functions instead of storing them?

