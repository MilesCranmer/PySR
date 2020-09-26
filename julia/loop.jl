using Distributed
const nprocs = 4
addprocs(4)
@everywhere include(".dataset_28330894764081783777.jl")
@everywhere include(".hyperparams_28330894764081783777.jl")
@everywhere include("sr.jl")


# 1. Start a population on every process
allPops = Future[]
bestSubPops = [Population(1) for j=1:nprocs]
hallOfFame = HallOfFame()

for i=1:nprocs
    npop=300
    future = @spawnat :any Population(npop, 3)
    push!(allPops, future)
end

npop=300
ncyclesperiteration=3000
fractionReplaced=0.1f0
verbosity=convert(Int, 1e9)
topn=10
niterations=10


# # 2. Start the cycle on every process:
for i=1:nprocs
    allPops[i] = @spawnat :any run(fetch(allPops[i]), ncyclesperiteration, verbosity=verbosity)
end
println("Started!")
cycles_complete = nprocs * 10
while cycles_complete > 0
    for i=1:nprocs
        if isready(allPops[i])
            cur_pop = fetch(allPops[i])
            bestSubPops[i] = bestSubPop(cur_pop, topn=topn)

            #Try normal copy...
            bestPops = Population([member for pop in bestSubPops for member in pop.members])

            for member in cur_pop.members
                size = countNodes(member.tree)
                if member.score < hallOfFame.members[size].score
                    hallOfFame.members[size] = deepcopy(member)
                    hallOfFame.exists[size] = true
                end
            end

            # Dominating pareto curve - must be better than all simpler equations
            dominating = PopMember[]
            open(hofFile, "w") do io
                debug(verbosity, "\n")
                debug(verbosity, "Hall of Fame:")
                debug(verbosity, "-----------------------------------------")
                debug(verbosity, "Complexity \t MSE \t Equation")
                println(io,"Complexity|MSE|Equation")
                for size=1:actualMaxsize
                    if hallOfFame.exists[size]
                        member = hallOfFame.members[size]
                        curMSE = MSE(evalTreeArray(member.tree), y)
                        numberSmallerAndBetter = sum([curMSE > MSE(evalTreeArray(hallOfFame.members[i].tree), y) for i=1:(size-1)])
                        betterThanAllSmaller = (numberSmallerAndBetter == 0)
                        if betterThanAllSmaller
                            debug(verbosity, "$size \t $(curMSE) \t $(stringTree(member.tree))")
                            println(io, "$size|$(curMSE)|$(stringTree(member.tree))")
                            push!(dominating, member)
                        end
                    end
                end
                debug(verbosity, "")
            end

            # Try normal copy otherwise.
            if migration
                for k in rand(1:npop, round(Integer, npop*fractionReplaced))
                    to_copy = rand(1:size(bestPops.members)[1])
                    cur_pop.members[k] = PopMember(
                        copyNode(bestPops.members[to_copy].tree),
                        bestPops.members[to_copy].score)
                end
            end

            if hofMigration && size(dominating)[1] > 0
                for k in rand(1:npop, round(Integer, npop*fractionReplacedHof))
                    # Copy in case one gets used twice
                    to_copy = rand(1:size(dominating)[1])
                    cur_pop.members[k] = PopMember(
                       copyNode(dominating[to_copy].tree)
                    )
                end
            end

            allPops[i] = @spawnat :any let
                tmp_pop = run(cur_pop, ncyclesperiteration, verbosity=verbosity)
                for j=1:tmp_pop.n
                    if rand() < 0.1
                        tmp_pop.members[j].tree = simplifyTree(tmp_pop.members[j].tree)
                        tmp_pop.members[j].tree = combineOperators(tmp_pop.members[j].tree)
                        if shouldOptimizeConstants
                            tmp_pop.members[j] = optimizeConstants(tmp_pop.members[j])
                        end
                    end
                end
                tmp_pop
            end

            global cycles_complete -= 1
        end
    end
    sleep(1e-3)
end

rmprocs(nprocs)


