using Distributed
addprocs(8)
@everywhere const nthreads = 8

@everywhere include("eureqa.jl")

println("Lets try to learn (x2^2 + cos(x3) + 5) using regularized evolution from scratch")
@everywhere const npop = 100
@everywhere const annealing = false
@everywhere const niterations = 30
@everywhere const ncyclesperiteration = 10000

# Generate random initial populations

# Create a mapping for running the algorithm on all processes
@everywhere f = (pop,)->run(pop, ncyclesperiteration, annealing)


allPops = [Population(npop, 3) for j=1:nthreads]
bestScore = Inf
# Repeat this many evolutions; we collect and migrate the best
# each time.
for k=1:4
    # Spawn independent evolutions
    futures = [@spawnat :any f(allPops[i]) for i=1:nthreads]

    # Gather them
    for i=1:nthreads
        allPops[i] = fetch(futures[i])
    end
    # Get best 10 models for each processes. Copy because we re-assign later.
    bestPops = deepcopy(Population([member for pop in allPops for member in bestSubPop(pop).members]))
    bestCurScoreIdx = argmin([bestPops.members[member].score for member=1:bestPops.n])
    bestCurScore = bestPops.members[bestCurScoreIdx].score
    println(bestCurScore, " is the score for ", stringTree(bestPops.members[bestCurScoreIdx].tree))

    # Migration
    for j=1:nthreads
        for k in rand(1:npop, 50)
            # Copy in case one gets copied twice
            allPops[j].members[k] = deepcopy(bestPops.members[rand(1:size(bestPops.members)[1])])
        end
    end
end





# julia> @everywhere include_string(Main, $(read("count_heads.jl", String)), "count_heads.jl")

# julia> a = @spawnat :any count_heads(100000000)
# Future(2, 1, 6, nothing)

# julia> b = @spawnat :any count_heads(100000000)
# Future(3, 1, 7, nothing)

# julia> fetch(a)+fetch(b)
# 100001564

# allPops = [Population(npop, 3) for j=1:nthreads]
# bestScore = Inf
# for i=1:10
    # tmpPops = fetch(pmap(f, allPops))
    # allPops[1:nthreads] = tmpPops[1:nthreads]
    # # Get best 11 models for each processes
    # bestPops = Population([member for pop in allPops for member in bestSubPop(pop).members])
    # bestCurScoreIdx = argmin([bestPops.members[member].score for member=1:bestPops.n])
    # bestCurScore = bestPops.members[bestCurScoreIdx].score
    # println(bestCurScore, " is the score for ", stringTree(bestPops.members[bestCurScoreIdx].tree))
# end


# function update(allPops::Array{Population, 1}, bestScore::Float64)
    # # Map it over our workers
    # #global allPops = deepcopy(pmap(f, deepcopy(allPops)))
    # #curAllPops = deepcopy(pmap(f, allPops))
    # curAllPops = pmap(f, allPops)
    # for j=1:nthreads
        # allPops[j] = curAllPops[j]
    # end

    # # Get best 10 models for each processes
    # bestPops = Population([member for pop in allPops for member in bestSubPop(pop).members])
    # bestCurScoreIdx = argmin([bestPops.members[member].score for member=1:bestPops.n])
    # bestCurScore = bestPops.members[bestCurScoreIdx].score
    # if bestCurScore < bestScore
        # bestScore = bestCurScore
        # println(bestScore, " is the score for ", stringTree(bestPops.members[bestCurScoreIdx].tree))
    # end

    # # Migration
    # for j=1:nthreads
        # allPops[j].members[1:50] = deepcopy(bestPops.members[rand(1:bestPops.n, 50)])
    # end
    # return allPops, bestScore
# end


# function runExperiment()
    # # Do niterations cycles
    # allPops = [Population(npop, 3) for j=1:nthreads]
    # bestScore = Inf
    # for i=1:niterations
        # allPops, bestScore = update(allPops, bestScore)
    # end

    # return bestScore
# end

# runExperiment()
