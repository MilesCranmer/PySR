# Running:

`julia paralleleureqa.jl`

## Modification

You can change the binary and unary operators in `eureqa.jl` here:
```
const binops = [plus, mult]
const unaops = [sin, cos, exp];
```

You can change the dataset here:
```
const nvar = 5;
const X = rand(100, nvar);
# Here is the function we want to learn (x2^2 + cos(x3) + 5)
const y = ((cx,)->cx^2).(X[:, 2]) + cos.(X[:, 3]) .+ 5.0;
```

The default number of processes is 10; this is set with
`addprocs(10)`.
