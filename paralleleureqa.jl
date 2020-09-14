using Distributed
addprocs(10)
@everywhere include("eureqa.jl")

println("Lets try to learn (x2^2 + cos(x3) + 5) using regularized evolution from scratch")
const npop = 100
const nthreads = 10
const annealing = false
bestScore = Inf
allPops = [Population(npop, 3) for i=1:nthreads]

@everywhere f = (pop,)->run(pop, 10000, annealing)
allPops = pmap(f, allPops)

for pop in allPops
    bestCurScoreIdx = argmin([pop.members[member].score for member=1:pop.n])
    bestCurScore = pop.members[bestCurScoreIdx].score
    if bestCurScore < bestScore
        global bestScore = bestCurScore
        println(bestScore, " is the score for ", stringTree(pop.members[bestCurScoreIdx].tree))
    end
end


