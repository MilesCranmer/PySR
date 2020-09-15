include("paralleleureqa.jl")
fullRun(1,
    npop=100,
    annealing=true,
    ncyclesperiteration=1000,
    fractionReplaced=0.1f0,
    verbosity=0)
@time for i=1:5
    fullRun(3,
        npop=100,
        annealing=true,
        ncyclesperiteration=100,
        fractionReplaced=0.1f0,
        verbosity=0
    )
end
