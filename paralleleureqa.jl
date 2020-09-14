using Distributed
const nthreads = 10
addprocs(nthreads)
@everywhere include("eureqa.jl")

println("Lets try to learn (x2^2 + cos(x3) + 5) using regularized evolution from scratch")
const npop = 100
const annealing = false
const niterations = 10
bestScore = Inf

# Generate random initial populations
allPops = [Population(npop, 3) for i=1:nthreads]

# Create a mapping for running the algorithm on all processes
@everywhere f = (pop,)->run(pop, 1000, annealing)

# Do niterations cycles
for i=1:niterations
    # Map it over our workers
    global allPops = deepcopy(pmap(f, allPops))

    # Get best 10 models for each processes
    bestPops = Population(vcat(map(((pop,)->bestSubPop(pop).members), allPops)...))
    for pop in bestPops
        bestCurScoreIdx = argmin([pop.members[member].score for member=1:pop.n])
        bestCurScore = pop.members[bestCurScoreIdx].score
        if bestCurScore < bestScore
            global bestScore = bestCurScore
            println(bestScore, " is the score for ", stringTree(pop.members[bestCurScoreIdx].tree))
        end
    end
    exit()
end


