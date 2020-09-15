include("eureqa.jl")

const nthreads = Threads.nthreads()
const migration = true
const hofMigration = true
const fractionReplacedHof = 0.05f0

# List of the best members seen all time
mutable struct HallOfFame
    members::Array{PopMember, 1}
    exists::Array{Bool, 1}

    # Arranged by complexity - store one at each.
    HallOfFame() = new([PopMember(Node(1f0), 1f9) for i=1:actualMaxsize], [false for i=1:actualMaxsize])
end


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
    hallOfFame = HallOfFame()

    for k=1:niterations
        # Spawn threads to run indepdent evolutions, then gather them
        @inbounds Threads.@threads for i=1:nthreads
            allPops[i] = run(allPops[i], ncyclesperiteration, annealing, verbosity=verbosity)
        end

        # Get best 10 models from each evolution. Copy because we re-assign later.
        bestPops = deepcopy(Population([member for pop in allPops for member in bestSubPop(pop).members]))

        #Update hall of fame
        for member in bestPops.members
            size = countNodes(member.tree)
            if member.score < hallOfFame.members[size].score
                hallOfFame.members[size] = deepcopy(member)
                hallOfFame.exists[size] = true
            end
        end

        dominating = PopMember[]
        debug(verbosity, "Hall of Fame:")
        debug(verbosity, "-----------------------------------------")
        debug(verbosity, "Complexity \t Score \t Equation")
        for size=1:maxsize
            if hallOfFame.exists[size]
                member = hallOfFame.members[size]
                numberSmallerAndBetter = sum([member.score > hallOfFame.members[i].score for i=1:(size-1)])
                betterThanAllSmaller = (numberSmallerAndBetter == 0)
                if betterThanAllSmaller
                    debug(verbosity, "$size \t $(member.score) \t $(stringTree(member.tree))")
                    push!(dominating, member)
                end
            end
        end
        debug(verbosity, "")

        # Migration
        if migration
            for j=1:nthreads
                for k in rand(1:npop, Integer(npop*fractionReplaced))
                    # Copy in case one gets used twice
                    allPops[j].members[k] = deepcopy(bestPops.members[rand(1:size(bestPops.members)[1])])
                end
            end
        end

        # Hall of fame migration
        if hofMigration
            for j=1:nthreads
                for k in rand(1:npop, Integer(npop*fractionReplacedHof))
                    # Copy in case one gets used twice
                    allPops[j].members[k] = deepcopy(dominating[rand(1:size(dominating)[1])])
                end
            end
        end

    end
end
