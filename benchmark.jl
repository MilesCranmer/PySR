include("paralleleureqa.jl")
using BenchmarkTools

fullRun(3,
    npop=100,
    annealing=true,
    ncyclesperiteration=100,
    fractionReplaced=0.1f0,
    verbosity=0)

t = @benchmark(fullRun(3,
        npop=100,
        annealing=true,
        ncyclesperiteration=100,
        fractionReplaced=0.1f0,
        verbosity=0
       ), evals=5)

tnoanneal = @benchmark(fullRun(3,
        npop=100,
        annealing=false,
        ncyclesperiteration=100,
        fractionReplaced=0.1f0,
        verbosity=0
       ), evals=5)

println("The median time is $(median(t)) with annealing, $(median(tnoanneal)) without")

