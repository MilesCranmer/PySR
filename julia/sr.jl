import Printf: @printf

function fullRun(niterations::Integer;
                npop::Integer=300,
                ncyclesperiteration::Integer=3000,
                fractionReplaced::Float32=0.1f0,
                verbosity::Integer=0,
                topn::Integer=10
               )

    testConfiguration()

    # 1. Start a population on every process
    allPops = Future[]
    # Set up a channel to send finished populations back to head node
    channels = [RemoteChannel(1) for j=1:npopulations]
    bestSubPops = [Population(1) for j=1:npopulations]
    hallOfFame = HallOfFame()
    frequencyComplexity = ones(Float32, actualMaxsize)
    curmaxsize = 3
    if warmupMaxsize == 0
        curmaxsize = maxsize
    end

    for i=1:npopulations
        future = @spawnat :any Population(npop, 3)
        push!(allPops, future)
    end

    # # 2. Start the cycle on every process:
    @sync for i=1:npopulations
        @async allPops[i] = @spawnat :any run(fetch(allPops[i]), ncyclesperiteration, curmaxsize, copy(frequencyComplexity)/sum(frequencyComplexity), verbosity=verbosity)
    end
    println("Started!")
    cycles_complete = npopulations * niterations
    if warmupMaxsize != 0
        curmaxsize += 1
        if curmaxsize > maxsize
            curmaxsize = maxsize
        end
    end

    last_print_time = time()
    num_equations = 0.0
    print_every_n_seconds = 5
    equation_speed = Float32[]

    for i=1:npopulations
        # Start listening for each population to finish:
        @async put!(channels[i], fetch(allPops[i]))
    end

    while cycles_complete > 0
        @inbounds for i=1:npopulations
            # Non-blocking check if a population is ready:
            if isready(channels[i])
                # Take the fetch operation from the channel since its ready
                cur_pop = take!(channels[i])
                bestSubPops[i] = bestSubPop(cur_pop, topn=topn)

                #Try normal copy...
                bestPops = Population([member for pop in bestSubPops for member in pop.members])

                for member in cur_pop.members
                    size = countNodes(member.tree)
                    frequencyComplexity[size] += 1
                    if member.score < hallOfFame.members[size].score
                        hallOfFame.members[size] = deepcopy(member)
                        hallOfFame.exists[size] = true
                    end
                end

                # Dominating pareto curve - must be better than all simpler equations
                dominating = PopMember[]
                open(hofFile, "w") do io
                    println(io,"Complexity|MSE|Equation")
                    for size=1:actualMaxsize
                        if hallOfFame.exists[size]
                            member = hallOfFame.members[size]
                            if weighted
                                curMSE = MSE(evalTreeArray(member.tree), y, weights)
                            else
                                curMSE = MSE(evalTreeArray(member.tree), y)
                            end
                            numberSmallerAndBetter = 0
                            for i=1:(size-1)
                                if weighted
                                    hofMSE = MSE(evalTreeArray(hallOfFame.members[i].tree), y, weights)
                                else
                                    hofMSE = MSE(evalTreeArray(hallOfFame.members[i].tree), y)
                                end
                                if (hallOfFame.exists[size] && curMSE > hofMSE)
                                    numberSmallerAndBetter += 1
                                end
                            end
                            betterThanAllSmaller = (numberSmallerAndBetter == 0)
                            if betterThanAllSmaller
                                println(io, "$size|$(curMSE)|$(stringTree(member.tree))")
                                push!(dominating, member)
                            end
                        end
                    end
                end
                cp(hofFile, hofFile*".bkup", force=true)

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

                @async begin
                    allPops[i] = @spawnat :any let
                        tmp_pop = run(cur_pop, ncyclesperiteration, curmaxsize, copy(frequencyComplexity)/sum(frequencyComplexity), verbosity=verbosity)
                        @inbounds @simd for j=1:tmp_pop.n
                            if rand() < 0.1
                                tmp_pop.members[j].tree = simplifyTree(tmp_pop.members[j].tree)
                                tmp_pop.members[j].tree = combineOperators(tmp_pop.members[j].tree)
                                if shouldOptimizeConstants
                                    tmp_pop.members[j] = optimizeConstants(tmp_pop.members[j])
                                end
                            end
                        end
                        tmp_pop = finalizeScores(tmp_pop)
                        tmp_pop
                    end
                    put!(channels[i], fetch(allPops[i]))
                end

                cycles_complete -= 1
                cycles_elapsed = npopulations * niterations - cycles_complete
                if warmupMaxsize != 0 && cycles_elapsed % warmupMaxsize == 0
                    curmaxsize += 1
                    if curmaxsize > maxsize
                        curmaxsize = maxsize
                    end
                end
                num_equations += ncyclesperiteration * npop / 10.0
            end
        end
        sleep(1e-3)
        elapsed = time() - last_print_time
        #Update if time has passed, and some new equations generated.
        if elapsed > print_every_n_seconds && num_equations > 0.0
            # Dominating pareto curve - must be better than all simpler equations
            current_speed = num_equations/elapsed
            average_over_m_measurements = 10 #for print_every...=5, this gives 50 second running average
            push!(equation_speed, current_speed)
            if length(equation_speed) > average_over_m_measurements
                deleteat!(equation_speed, 1)
            end
            average_speed = sum(equation_speed)/length(equation_speed)
            curMSE = baselineMSE
            lastMSE = curMSE
            lastComplexity = 0
            if verbosity > 0
                @printf("\n")
                @printf("Cycles per second: %.3e\n", round(average_speed, sigdigits=3))
                cycles_elapsed = npopulations * niterations - cycles_complete
                @printf("Progress: %d / %d total iterations (%.3f%%)\n", cycles_elapsed, npopulations * niterations, 100.0*cycles_elapsed/(npopulations*niterations))
                @printf("Hall of Fame:\n")
                @printf("-----------------------------------------\n")
                @printf("%-10s  %-8s   %-8s  %-8s\n", "Complexity", "MSE", "Score", "Equation")
                @printf("%-10d  %-8.3e  %-8.3e  %-.f\n", 0, curMSE, 0f0, avgy)
            end

            for size=1:actualMaxsize
                if hallOfFame.exists[size]
                    member = hallOfFame.members[size]
                    if weighted
                        curMSE = MSE(evalTreeArray(member.tree), y, weights)
                    else
                        curMSE = MSE(evalTreeArray(member.tree), y)
                    end
                    numberSmallerAndBetter = 0
                    for i=1:(size-1)
                        if weighted
                            hofMSE = MSE(evalTreeArray(hallOfFame.members[i].tree), y, weights)
                        else
                            hofMSE = MSE(evalTreeArray(hallOfFame.members[i].tree), y)
                        end
                        if (hallOfFame.exists[size] && curMSE > hofMSE)
                            numberSmallerAndBetter += 1
                        end
                    end
                    betterThanAllSmaller = (numberSmallerAndBetter == 0)
                    if betterThanAllSmaller
                        delta_c = size - lastComplexity
                        delta_l_mse = log(curMSE/lastMSE)
                        score = convert(Float32, -delta_l_mse/delta_c)
                        if verbosity > 0
                            @printf("%-10d  %-8.3e  %-8.3e  %-s\n" , size, curMSE, score, stringTree(member.tree))
                        end
                        lastMSE = curMSE
                        lastComplexity = size
                    end
                end
            end
            debug(verbosity, "")
            last_print_time = time()
            num_equations = 0.0
        end
    end
end
