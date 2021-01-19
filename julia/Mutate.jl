# Go through one mutation cycle
function iterate(member::PopMember, T::Float32, curmaxsize::Integer, frequencyComplexity::Array{Float32, 1})::PopMember
    prev = member.tree
    tree = prev
    #TODO - reconsider this
    if batching
        beforeLoss = scoreFuncBatch(prev)
    else
        beforeLoss = member.score
    end

    mutationChoice = rand()
    #More constants => more likely to do constant mutation
    weightAdjustmentMutateConstant = min(8, countConstants(prev))/8.0
    cur_weights = copy(mutationWeights) .* 1.0
    cur_weights[1] *= weightAdjustmentMutateConstant
    n = countNodes(prev)
    depth = countDepth(prev)

    # If equation too big, don't add new operators
    if n >= curmaxsize || depth >= maxdepth
        cur_weights[3] = 0.0
        cur_weights[4] = 0.0
    end
    cur_weights /= sum(cur_weights)
    cweights = cumsum(cur_weights)

    successful_mutation = false
    #TODO: Currently we dont take this \/ into account
    is_success_always_possible = true
    attempts = 0
    max_attempts = 10

    #############################################
    # Mutations
    #############################################
    while (!successful_mutation) && attempts < max_attempts
        tree = copyNode(prev)
        successful_mutation = true
        if mutationChoice < cweights[1]
            tree = mutateConstant(tree, T)

            is_success_always_possible = true
            # Mutating a constant shouldn't invalidate an already-valid function

        elseif mutationChoice < cweights[2]
            tree = mutateOperator(tree)

            is_success_always_possible = true
            # Can always mutate to the same operator

        elseif mutationChoice < cweights[3]
            if rand() < 0.5
                tree = appendRandomOp(tree)
            else
                tree = prependRandomOp(tree)
            end
            is_success_always_possible = false
            # Can potentially have a situation without success
        elseif mutationChoice < cweights[4]
            tree = insertRandomOp(tree)
            is_success_always_possible = false
        elseif mutationChoice < cweights[5]
            tree = deleteRandomOp(tree)
            is_success_always_possible = true
        elseif mutationChoice < cweights[6]
            tree = simplifyTree(tree) # Sometimes we simplify tree
            tree = combineOperators(tree) # See if repeated constants at outer levels
            return PopMember(tree, beforeLoss)

            is_success_always_possible = true
            # Simplification shouldn't hurt complexity; unless some non-symmetric constraint
            # to commutative operator...

        elseif mutationChoice < cweights[7]
            tree = genRandomTree(5) # Sometimes we generate a new tree completely tree

            is_success_always_possible = true
        else # no mutation applied
            return PopMember(tree, beforeLoss)
        end

        # Check for illegal equations
        for i=1:nbin
            if successful_mutation && flagBinOperatorComplexity(tree, i)
                successful_mutation = false
            end
        end
        for i=1:nuna
            if successful_mutation && flagUnaOperatorComplexity(tree, i)
                successful_mutation = false
            end
        end

        attempts += 1
    end
    #############################################

    if !successful_mutation
        return PopMember(copyNode(prev), beforeLoss)
    end

    if batching
        afterLoss = scoreFuncBatch(tree)
    else
        afterLoss = scoreFunc(tree)
    end

    if annealing
        delta = afterLoss - beforeLoss
        probChange = exp(-delta/(T*alpha))
        if useFrequency
            oldSize = countNodes(prev)
            newSize = countNodes(tree)
            probChange *= frequencyComplexity[oldSize] / frequencyComplexity[newSize]
        end

        return_unaltered = (isnan(afterLoss) || probChange < rand())
        if return_unaltered
            return PopMember(copyNode(prev), beforeLoss)
        end
    end
    return PopMember(tree, afterLoss)
end
