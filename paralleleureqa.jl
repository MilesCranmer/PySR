include("eureqa.jl")

const nthreads = Threads.nthreads()

function fullRun(niterations::Integer;
                npop::Integer=300,
                annealing::Bool=true,
                ncyclesperiteration::Integer=3000,
                fractionReplaced::Float32=0.1f0,
                verbosity::Integer=0,
               )
    debug(verbosity, "Lets try to learn (x2^2 + cos(x3)) using regularized evolution from scratch")
    debug(verbosity, "Running with $nthreads threads")
    # Generate random initial populations
    allPops = [Population(npop, 3) for j=1:nthreads]
    # Repeat this many evolutions; we collect and migrate the best
    # each time.
    for k=1:niterations
        # Spawn threads to run indepdent evolutions, then gather them
        @inbounds Threads.@threads for i=1:nthreads
            allPops[i] = run(allPops[i], ncyclesperiteration, annealing, verbosity=verbosity)
        end

        # Get best 10 models from each evolution. Copy because we re-assign later.
        bestPops = deepcopy(Population([member for pop in allPops for member in bestSubPop(pop).members]))
        bestCurScoreIdx = argmin([bestPops.members[member].score for member=1:bestPops.n])
        bestCurScore = bestPops.members[bestCurScoreIdx].score
        debug(verbosity, bestCurScore, " is the score for ", stringTree(bestPops.members[bestCurScoreIdx].tree))

        # Migration
        for j=1:nthreads
            for k in rand(1:npop, Integer(npop*fractionReplaced))
                # Copy in case one gets used twice
                allPops[j].members[k] = deepcopy(bestPops.members[rand(1:size(bestPops.members)[1])])
            end
        end
    end
end

