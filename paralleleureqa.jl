include("eureqa.jl")

println("Lets try to learn (x2^2 + cos(x3)) using regularized evolution from scratch")
const nthreads = Threads.nthreads()
println("Running with $nthreads threads")
const npop = 1000
const annealing = true
const niterations = 100
const ncyclesperiteration = 30000

# Generate random initial populations
allPops = [Population(npop, 3) for j=1:nthreads]
bestScore = Inf
# Repeat this many evolutions; we collect and migrate the best
# each time.
for k=1:niterations

    # Spawn threads to run indepdent evolutions, then gather them
    @inbounds Threads.@threads for i=1:nthreads
        allPops[i] = run(allPops[i], ncyclesperiteration, annealing, verbose=500)
    end

    # Get best 10 models from each evolution. Copy because we re-assign later.
    bestPops = deepcopy(Population([member for pop in allPops for member in bestSubPop(pop).members]))
    bestCurScoreIdx = argmin([bestPops.members[member].score for member=1:bestPops.n])
    bestCurScore = bestPops.members[bestCurScoreIdx].score
    println(bestCurScore, " is the score for ", stringTree(bestPops.members[bestCurScoreIdx].tree))

    # Migration
    for j=1:nthreads
        for k in rand(1:npop, 50)
            # Copy in case one gets used twice
            allPops[j].members[k] = deepcopy(bestPops.members[rand(1:size(bestPops.members)[1])])
        end
    end
end

