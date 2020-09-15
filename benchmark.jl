include("paralleleureqa.jl")
fullRun(1,
    npop=100,
    annealing=true,
    ncyclesperiteration=1000,
    fractionReplaced=0.1f0,
    verbosity=0)
@time fullRun(3,
    npop=100,
    annealing=true,
    ncyclesperiteration=1000,
    fractionReplaced=0.1f0,
    verbosity=0
)
