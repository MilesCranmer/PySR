using Distributed
const nthreads = 8
addprocs(nthreads)

@everywhere include("eureqa.jl")

println("Lets try to learn (x2^2 + cos(x3) + 5) using regularized evolution from scratch")
const npop = 100
const annealing = false
const niterations = 10
const ncyclesperiteration = 1000

# Generate random initial populations

# Create a mapping for running the algorithm on all processes
@everywhere f = (pop,)->run(pop, ncyclesperiteration, annealing)


function update(allPops::Array{Population, 1}, bestScore::Float64, pool::AbstractWorkerPool)
    # Map it over our workers
    #global allPops = deepcopy(pmap(f, deepcopy(allPops)))
    curAllPops = deepcopy(pmap(f, allPops))
    for j=1:nthreads
        allPops[j] = curAllPops[j]
    end

    # Get best 10 models for each processes
    bestPops = Population([member for pop in allPops for member in bestSubPop(pop).members])
    bestCurScoreIdx = argmin([bestPops.members[member].score for member=1:bestPops.n])
    bestCurScore = bestPops.members[bestCurScoreIdx].score
    if bestCurScore < bestScore
        bestScore = bestCurScore
        println(bestScore, " is the score for ", stringTree(bestPops.members[bestCurScoreIdx].tree))
    end

    # Migration
    for j=1:nthreads
        allPops[j].members[1:50] = deepcopy(bestPops.members[rand(1:bestPops.n, 50)])
    end
    return allPops, bestScore
end


function runExperiment()
    # Do niterations cycles
    allPops = [Population(npop, 3) for j=1:nthreads]
    bestScore = Inf
    #pool = CachingPool(workers())
    pool = WorkerPool(workers())
    for i=1:niterations
        allPops, bestScore = update(allPops, bestScore, pool)
    end

    return bestScore
end

runExperiment()

