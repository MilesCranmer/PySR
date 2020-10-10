import Optim
import Printf: @printf
import Random: shuffle!, randperm

const maxdegree = 2
const actualMaxsize = maxsize + maxdegree


# Sum of square error between two arrays
function SSE(x::Array{Float32}, y::Array{Float32})::Float32
    diff = (x - y)
    return sum(diff .* diff)
end
function SSE(x::Nothing, y::Array{Float32})::Float32
    return 1f9
end

# Sum of square error between two arrays, with weights
function SSE(x::Array{Float32}, y::Array{Float32}, w::Array{Float32})::Float32
    diff = (x - y)
    return sum(diff .* diff .* w)
end
function SSE(x::Nothing, y::Array{Float32}, w::Array{Float32})::Float32
    return Nothing
end

# Mean of square error between two arrays
function MSE(x::Nothing, y::Array{Float32})::Float32
    return 1f9
end

# Mean of square error between two arrays
function MSE(x::Array{Float32}, y::Array{Float32})::Float32
    return SSE(x, y)/size(x)[1]
end

# Mean of square error between two arrays
function MSE(x::Nothing, y::Array{Float32}, w::Array{Float32})::Float32
    return 1f9
end

# Mean of square error between two arrays
function MSE(x::Array{Float32}, y::Array{Float32}, w::Array{Float32})::Float32
    return SSE(x, y, w)/sum(w)
end

const len = size(X)[1]

if weighted
    const avgy = sum(y .* weights)/sum(weights)
    const baselineMSE = MSE(y, convert(Array{Float32, 1}, ones(len) .* avgy), weights)
else
    const avgy = sum(y)/len
    const baselineMSE = MSE(y, convert(Array{Float32, 1}, ones(len) .* avgy))
end


id = (x,) -> x
const nuna = size(unaops)[1]
const nbin = size(binops)[1]
const nops = nuna + nbin
const nvar = size(X)[2];

function debug(verbosity, string...)
    verbosity > 0 ? println(string...) : nothing
end

function getTime()::Integer
    return round(Integer, 1e3*(time()-1.6e9))
end

# Define a serialization format for the symbolic equations:
mutable struct Node
    #Holds operators, variables, constants in a tree
    degree::Integer #0 for constant/variable, 1 for cos/sin, 2 for +/* etc.
    val::Union{Float32, Integer} #Either const value, or enumerates variable
    constant::Bool #false if variable
    op::Integer #enumerates operator (separately for degree=1,2)
    l::Union{Node, Nothing}
    r::Union{Node, Nothing}

    Node(val::Float32) = new(0, val, true, 1, nothing, nothing)
    Node(val::Integer) = new(0, val, false, 1, nothing, nothing)
    Node(op::Integer, l::Node) = new(1, 0.0f0, false, op, l, nothing)
    Node(op::Integer, l::Union{Float32, Integer}) = new(1, 0.0f0, false, op, Node(l), nothing)
    Node(op::Integer, l::Node, r::Node) = new(2, 0.0f0, false, op, l, r)

    #Allow to pass the leaf value without additional node call:
    Node(op::Integer, l::Union{Float32, Integer}, r::Node) = new(2, 0.0f0, false, op, Node(l), r)
    Node(op::Integer, l::Node, r::Union{Float32, Integer}) = new(2, 0.0f0, false, op, l, Node(r))
    Node(op::Integer, l::Union{Float32, Integer}, r::Union{Float32, Integer}) = new(2, 0.0f0, false, op, Node(l), Node(r))
end

# Copy an equation (faster than deepcopy)
function copyNode(tree::Node)::Node
   if tree.degree == 0
       return Node(tree.val)
   elseif tree.degree == 1
       return Node(tree.op, copyNode(tree.l))
    else
        return Node(tree.op, copyNode(tree.l), copyNode(tree.r))
   end
end

# Count the operators, constants, variables in an equation
function countNodes(tree::Node)::Integer
    if tree.degree == 0
        return 1
    elseif tree.degree == 1
        return 1 + countNodes(tree.l)
    else
        return 1 + countNodes(tree.l) + countNodes(tree.r)
    end
end

# Count the max depth of a tree
function countDepth(tree::Node)::Integer
    if tree.degree == 0
        return 1
    elseif tree.degree == 1
        return 1 + countDepth(tree.l)
    else
        return 1 + max(countDepth(tree.l), countDepth(tree.r))
    end
end

# Convert an equation to a string
function stringTree(tree::Node)::String
    if tree.degree == 0
        if tree.constant
            return string(tree.val)
        else
            if useVarMap
                return varMap[tree.val]
            else
                return "x$(tree.val - 1)"
            end
        end
    elseif tree.degree == 1
        return "$(unaops[tree.op])($(stringTree(tree.l)))"
    else
        return "$(binops[tree.op])($(stringTree(tree.l)), $(stringTree(tree.r)))"
    end
end

# Print an equation
function printTree(tree::Node)
    println(stringTree(tree))
end

# Return a random node from the tree
function randomNode(tree::Node)::Node
    if tree.degree == 0
        return tree
    end
    a = countNodes(tree)
    b = 0
    c = 0
    if tree.degree >= 1
        b = countNodes(tree.l)
    end
    if tree.degree == 2
        c = countNodes(tree.r)
    end

    i = rand(1:1+b+c)
    if i <= b
        return randomNode(tree.l)
    elseif i == b + 1
        return tree
    end

    return randomNode(tree.r)
end

# Count the number of unary operators in the equation
function countUnaryOperators(tree::Node)::Integer
    if tree.degree == 0
        return 0
    elseif tree.degree == 1
        return 1 + countUnaryOperators(tree.l)
    else
        return 0 + countUnaryOperators(tree.l) + countUnaryOperators(tree.r)
    end
end

# Count the number of binary operators in the equation
function countBinaryOperators(tree::Node)::Integer
    if tree.degree == 0
        return 0
    elseif tree.degree == 1
        return 0 + countBinaryOperators(tree.l)
    else
        return 1 + countBinaryOperators(tree.l) + countBinaryOperators(tree.r)
    end
end

# Count the number of operators in the equation
function countOperators(tree::Node)::Integer
    return countUnaryOperators(tree) + countBinaryOperators(tree)
end

# Randomly convert an operator into another one (binary->binary;
# unary->unary)
function mutateOperator(tree::Node)::Node
    if countOperators(tree) == 0
        return tree
    end
    node = randomNode(tree)
    while node.degree == 0
        node = randomNode(tree)
    end
    if node.degree == 1
        node.op = rand(1:length(unaops))
    else
        node.op = rand(1:length(binops))
    end
    return tree
end

# Count the number of constants in an equation
function countConstants(tree::Node)::Integer
    if tree.degree == 0
        return convert(Integer, tree.constant)
    elseif tree.degree == 1
        return 0 + countConstants(tree.l)
    else
        return 0 + countConstants(tree.l) + countConstants(tree.r)
    end
end

# Randomly perturb a constant
function mutateConstant(
        tree::Node, T::Float32,
        probNegate::Float32=0.01f0)::Node
    # T is between 0 and 1.

    if countConstants(tree) == 0
        return tree
    end
    node = randomNode(tree)
    while node.degree != 0 || node.constant == false
        node = randomNode(tree)
    end

    bottom = 0.1f0
    maxChange = perturbationFactor * T + 1.0f0 + bottom
    factor = maxChange^Float32(rand())
    makeConstBigger = rand() > 0.5

    if makeConstBigger
        node.val *= factor
    else
        node.val /= factor
    end

    if rand() > probNegate
        node.val *= -1
    end

    return tree
end

# Evaluate an equation over an array of datapoints
function evalTreeArray(tree::Node)::Union{Array{Float32, 1}, Nothing}
    return evalTreeArray(tree, X)
end


# Evaluate an equation over an array of datapoints
function evalTreeArray(tree::Node, cX::Array{Float32, 2})::Union{Array{Float32, 1}, Nothing}
    clen = size(cX)[1]
    if tree.degree == 0
        if tree.constant
            return fill(tree.val, clen)
        else
            return copy(cX[:, tree.val])
        end
    elseif tree.degree == 1
        cumulator = evalTreeArray(tree.l, cX)
        if cumulator == nothing
            return nothing
        end
        op = unaops[tree.op]
        @fastmath @inbounds @simd for i=1:clen
            cumulator[i] = op(cumulator[i])
        end
        @inbounds @simd for i=1:clen
            if isinf(cumulator[i]) || isnan(cumulator[i])
                return nothing
            end
        end
        return cumulator
    else
        op = binops[tree.op]
        cumulator = evalTreeArray(tree.l, cX)
        if cumulator == nothing
            return nothing
        end
        array2 = evalTreeArray(tree.r, cX)
        if array2 == nothing
            return nothing
        end

        @fastmath @inbounds @simd for i=1:clen
            cumulator[i] = op(cumulator[i], array2[i])
        end
        @inbounds @simd for i=1:clen
            if isinf(cumulator[i]) || isnan(cumulator[i])
                return nothing
            end
        end
        return cumulator
    end
end

# Score an equation
function scoreFunc(tree::Node)::Float32
    prediction = evalTreeArray(tree)
    if prediction == nothing
        return 1f9
    end
    if weighted
        mse = MSE(prediction, y, weights)
    else
        mse = MSE(prediction, y)
    end
    return mse / baselineMSE + countNodes(tree)*parsimony
end

# Score an equation with a small batch
function scoreFuncBatch(tree::Node)::Float32
    # batchSize
    batch_idx = randperm(len)[1:batchSize]
    batch_X = X[batch_idx, :]
    prediction = evalTreeArray(tree, batch_X)
    if prediction == nothing
        return 1f9
    end
    size_adjustment = 1f0
    batch_y = y[batch_idx]
    if weighted
        batch_w = weights[batch_idx]
        mse = MSE(prediction, batch_y, batch_w)
        size_adjustment = 1f0 * len / batchSize
    else
        mse = MSE(prediction, batch_y)
    end
    return size_adjustment * mse / baselineMSE + countNodes(tree)*parsimony
end

# Add a random unary/binary operation to the end of a tree
function appendRandomOp(tree::Node)::Node
    node = randomNode(tree)
    while node.degree != 0
        node = randomNode(tree)
    end

    choice = rand()
    makeNewBinOp = choice < nbin/nops
    if rand() > 0.5
        left = Float32(randn())
    else
        left = rand(1:nvar)
    end
    if rand() > 0.5
        right = Float32(randn())
    else
        right = rand(1:nvar)
    end

    if makeNewBinOp
        newnode = Node(
            rand(1:length(binops)),
            left,
            right
        )
    else
        newnode = Node(
            rand(1:length(unaops)),
            left
        )
    end
    node.l = newnode.l
    node.r = newnode.r
    node.op = newnode.op
    node.degree = newnode.degree
    node.val = newnode.val
    node.constant = newnode.constant
    return tree
end

# Insert random node
function insertRandomOp(tree::Node)::Node
    node = randomNode(tree)
    choice = rand()
    makeNewBinOp = choice < nbin/nops
    left = copyNode(node)

    if makeNewBinOp
        right = randomConstantNode()
        newnode = Node(
            rand(1:length(binops)),
            left,
            right
        )
    else
        newnode = Node(
            rand(1:length(unaops)),
            left
        )
    end
    node.l = newnode.l
    node.r = newnode.r
    node.op = newnode.op
    node.degree = newnode.degree
    node.val = newnode.val
    node.constant = newnode.constant
    return tree
end

# Add random node to the top of a tree
function prependRandomOp(tree::Node)::Node
    node = tree
    choice = rand()
    makeNewBinOp = choice < nbin/nops
    left = copyNode(tree)

    if makeNewBinOp
        right = randomConstantNode()
        newnode = Node(
            rand(1:length(binops)),
            left,
            right
        )
    else
        newnode = Node(
            rand(1:length(unaops)),
            left
        )
    end
    node.l = newnode.l
    node.r = newnode.r
    node.op = newnode.op
    node.degree = newnode.degree
    node.val = newnode.val
    node.constant = newnode.constant
    return node
end

function randomConstantNode()::Node
    if rand() > 0.5
        val = Float32(randn())
    else
        val = rand(1:nvar)
    end
    newnode = Node(val)
    return newnode
end

# Return a random node from the tree with parent
function randomNodeAndParent(tree::Node, parent::Union{Node, Nothing})::Tuple{Node, Union{Node, Nothing}}
    if tree.degree == 0
        return tree, parent
    end
    a = countNodes(tree)
    b = 0
    c = 0
    if tree.degree >= 1
        b = countNodes(tree.l)
    end
    if tree.degree == 2
        c = countNodes(tree.r)
    end

    i = rand(1:1+b+c)
    if i <= b
        return randomNodeAndParent(tree.l, tree)
    elseif i == b + 1
        return tree, parent
    end

    return randomNodeAndParent(tree.r, tree)
end

# Select a random node, and replace it an the subtree
# with a variable or constant
function deleteRandomOp(tree::Node)::Node
    node, parent = randomNodeAndParent(tree, nothing)
    isroot = (parent == nothing)

    if node.degree == 0
        # Replace with new constant
        newnode = randomConstantNode()
        node.l = newnode.l
        node.r = newnode.r
        node.op = newnode.op
        node.degree = newnode.degree
        node.val = newnode.val
        node.constant = newnode.constant
    elseif node.degree == 1
        # Join one of the children with the parent
        if isroot
            return node.l
        elseif parent.l == node
            parent.l = node.l
        else
            parent.r = node.l
        end
    else
        # Join one of the children with the parent
        if rand() < 0.5
            if isroot
                return node.l
            elseif parent.l == node
                parent.l = node.l
            else
                parent.r = node.l
            end
        else
            if isroot
                return node.r
            elseif parent.l == node
                parent.l = node.r
            else
                parent.r = node.r
            end
        end
    end
    return tree
end

# Simplify tree
function combineOperators(tree::Node)::Node
    # (const (+*) const) already accounted for
    # ((const + var) + const) => (const + var)
    # ((const * var) * const) => (const * var)
    # (anything commutative!)
    if tree.degree == 2 && (binops[tree.op] == plus || binops[tree.op] == mult)
        op = tree.op
        if tree.l.constant || tree.r.constant
            # Put the constant in r
            if tree.l.constant
                tmp = tree.r
                tree.r = tree.l
                tree.l = tmp
            end
            topconstant = tree.r.val
            # Simplify down first
            tree.l = combineOperators(tree.l)
            below = tree.l
            if below.degree == 2 && below.op == op
                if below.l.constant
                    tree = below
                    tree.l.val = binops[op](tree.l.val, topconstant)
                elseif below.r.constant
                    tree = below
                    tree.r.val = binops[op](tree.r.val, topconstant)
                end
            end
        end
    end
    return tree
end

# Simplify tree
function simplifyTree(tree::Node)::Node
    if tree.degree == 1
        tree.l = simplifyTree(tree.l)
        if tree.l.degree == 0 && tree.l.constant
            return Node(unaops[tree.op](tree.l.val))
        end
    elseif tree.degree == 2
        tree.l = simplifyTree(tree.l)
        tree.r = simplifyTree(tree.r)
        constantsBelow = (
             tree.l.degree == 0 && tree.l.constant &&
             tree.r.degree == 0 && tree.r.constant
        )
        if constantsBelow
            return Node(binops[tree.op](tree.l.val, tree.r.val))
        end
    end
    return tree
end

# Define a member of population by equation, score, and age
mutable struct PopMember
    tree::Node
    score::Float32
    birth::Integer

    PopMember(t::Node) = new(t, scoreFunc(t), getTime())
    PopMember(t::Node, score::Float32) = new(t, score, getTime())

end

# Go through one simulated annealing mutation cycle
#  exp(-delta/T) defines probability of accepting a change
function iterate(member::PopMember, T::Float32)::PopMember
    prev = member.tree
    tree = copyNode(prev)
    #TODO - reconsider this
    if batching
        beforeLoss = scoreFuncBatch(member.tree)
    else
        beforeLoss = member.score
    end

    mutationChoice = rand()
    weightAdjustmentMutateConstant = min(8, countConstants(tree))/8.0
    cur_weights = copy(mutationWeights) .* 1.0
    cur_weights[1] *= weightAdjustmentMutateConstant
    cur_weights /= sum(cur_weights)
    cweights = cumsum(cur_weights)
    n = countNodes(tree)
    depth = countDepth(tree)

    if mutationChoice < cweights[1]
        tree = mutateConstant(tree, T)
    elseif mutationChoice < cweights[2]
        tree = mutateOperator(tree)
    elseif mutationChoice < cweights[3] && n < maxsize && depth < maxdepth
        if rand() < 0.5
            tree = appendRandomOp(tree)
        else
            tree = prependRandomOp(tree)
        end
    elseif mutationChoice < cweights[4] && n < maxsize && depth < maxdepth
        tree = insertRandomOp(tree)
    elseif mutationChoice < cweights[5]
        tree = deleteRandomOp(tree)
    elseif mutationChoice < cweights[6]
        tree = simplifyTree(tree) # Sometimes we simplify tree
        tree = combineOperators(tree) # See if repeated constants at outer levels
        return PopMember(tree, beforeLoss)
    elseif mutationChoice < cweights[7]
        tree = genRandomTree(5) # Sometimes we generate a new tree completely tree
    else
        return PopMember(tree, beforeLoss)
    end

    if batching
        afterLoss = scoreFuncBatch(tree)
    else
        afterLoss = scoreFunc(tree)
    end

    if annealing
        delta = afterLoss - beforeLoss
        probChange = exp(-delta/(T*alpha))

        return_unaltered = (isnan(afterLoss) || probChange < rand())
        if return_unaltered
            return PopMember(copyNode(prev), beforeLoss)
        end
    end
    return PopMember(tree, afterLoss)
end

# Create a random equation by appending random operators
function genRandomTree(length::Integer)::Node
    tree = Node(1.0f0)
    for i=1:length
        tree = appendRandomOp(tree)
    end
    return tree
end


# A list of members of the population, with easy constructors,
#  which allow for random generation of new populations
mutable struct Population
    members::Array{PopMember, 1}
    n::Integer

    Population(pop::Array{PopMember, 1}) = new(pop, size(pop)[1])
    Population(npop::Integer) = new([PopMember(genRandomTree(3)) for i=1:npop], npop)
    Population(npop::Integer, nlength::Integer) = new([PopMember(genRandomTree(nlength)) for i=1:npop], npop)

end

# Sample 10 random members of the population, and make a new one
function samplePop(pop::Population)::Population
    idx = rand(1:pop.n, ns)
    return Population(pop.members[idx])
end

# Sample the population, and get the best member from that sample
function bestOfSample(pop::Population)::PopMember
    sample = samplePop(pop)
    best_idx = argmin([sample.members[member].score for member=1:sample.n])
    return sample.members[best_idx]
end

function finalizeScores(pop::Population)::Population
    need_recalculate = batching
    if need_recalculate
        @inbounds @simd for member=1:pop.n
            pop.members[member].score = scoreFunc(pop.members[member].tree)
        end
    end
    return pop
end

# Return best 10 examples
function bestSubPop(pop::Population; topn::Integer=10)::Population
    best_idx = sortperm([pop.members[member].score for member=1:pop.n])
    return Population(pop.members[best_idx[1:topn]])
end

# Pass through the population several times, replacing the oldest
# with the fittest of a small subsample
function regEvolCycle(pop::Population, T::Float32)::Population
    # Batch over each subsample. Can give 15% improvement in speed; probably moreso for large pops.
    # but is ultimately a different algorithm than regularized evolution, and might not be
    # as good.
    if fast_cycle
        shuffle!(pop.members)
        n_evol_cycles = round(Integer, pop.n/ns)
        babies = Array{PopMember}(undef, n_evol_cycles)

        # Iterate each ns-member sub-sample
        @inbounds Threads.@threads for i=1:n_evol_cycles
            best_score = Inf32
            best_idx = 1+(i-1)*ns
            # Calculate best member of the subsample:
            for sub_i=1+(i-1)*ns:i*ns
                if pop.members[sub_i].score < best_score
                    best_score = pop.members[sub_i].score
                    best_idx = sub_i
                end
            end
            allstar = pop.members[best_idx]
            babies[i] = iterate(allstar, T)
        end

        # Replace the n_evol_cycles-oldest members of each population
        @inbounds for i=1:n_evol_cycles
            oldest = argmin([pop.members[member].birth for member=1:pop.n])
            pop.members[oldest] = babies[i]
        end
    else
        for i=1:round(Integer, pop.n/ns)
            allstar = bestOfSample(pop)
            baby = iterate(allstar, T)
            #printTree(baby.tree)
            oldest = argmin([pop.members[member].birth for member=1:pop.n])
            pop.members[oldest] = baby
        end
    end

    return pop
end

# Cycle through regularized evolution many times,
# printing the fittest equation every 10% through
function run(
        pop::Population,
        ncycles::Integer;
        verbosity::Integer=0
        )::Population

    allT = LinRange(1.0f0, 0.0f0, ncycles)
    for iT in 1:size(allT)[1]
        if annealing
            pop = regEvolCycle(pop, allT[iT])
        else
            pop = regEvolCycle(pop, 1.0f0)
        end

        if verbosity > 0 && (iT % verbosity == 0)
            bestPops = bestSubPop(pop)
            bestCurScoreIdx = argmin([bestPops.members[member].score for member=1:bestPops.n])
            bestCurScore = bestPops.members[bestCurScoreIdx].score
            debug(verbosity, bestCurScore, " is the score for ", stringTree(bestPops.members[bestCurScoreIdx].tree))
        end
    end

    return pop
end

# Get all the constants from a tree
function getConstants(tree::Node)::Array{Float32, 1}
    if tree.degree == 0
        if tree.constant
            return [tree.val]
        else
            return Float32[]
        end
    elseif tree.degree == 1
        return getConstants(tree.l)
    else
        both = [getConstants(tree.l), getConstants(tree.r)]
        return [constant for subtree in both for constant in subtree]
    end
end

# Set all the constants inside a tree
function setConstants(tree::Node, constants::Array{Float32, 1})
    if tree.degree == 0
        if tree.constant
            tree.val = constants[1]
        end
    elseif tree.degree == 1
        setConstants(tree.l, constants)
    else
        numberLeft = countConstants(tree.l)
        setConstants(tree.l, constants)
        setConstants(tree.r, constants[numberLeft+1:end])
    end
end


# Proxy function for optimization
function optFunc(x::Array{Float32, 1}, tree::Node)::Float32
    setConstants(tree, x)
    return scoreFunc(tree)
end

# Use Nelder-Mead to optimize the constants in an equation
function optimizeConstants(member::PopMember)::PopMember
    nconst = countConstants(member.tree)
    if nconst == 0
        return member
    end
    x0 = getConstants(member.tree)
    f(x::Array{Float32,1})::Float32 = optFunc(x, member.tree)
    if size(x0)[1] == 1
        algorithm = Optim.Newton
    else
        algorithm = Optim.NelderMead
    end

    try
        result = Optim.optimize(f, x0, algorithm(), Optim.Options(iterations=100))
        # Try other initial conditions:
        for i=1:nrestarts
            tmpresult = Optim.optimize(f, x0 .* (1f0 .+ 5f-1*randn(Float32, size(x0)[1])), algorithm(), Optim.Options(iterations=100))
            if tmpresult.minimum < result.minimum
                result = tmpresult
            end
        end

        if Optim.converged(result)
            setConstants(member.tree, result.minimizer)
            member.score = convert(Float32, result.minimum)
            member.birth = getTime()
        else
            setConstants(member.tree, x0)
        end
    catch error
        # Fine if optimization encountered domain error, just return x0
        if isa(error, AssertionError)
            setConstants(member.tree, x0)
        else
            throw(error)
        end
    end
    return member
end


# List of the best members seen all time
mutable struct HallOfFame
    members::Array{PopMember, 1}
    exists::Array{Bool, 1} #Whether it has been set

    # Arranged by complexity - store one at each.
    HallOfFame() = new([PopMember(Node(1f0), 1f9) for i=1:actualMaxsize], [false for i=1:actualMaxsize])
end


# Check for errors before they happen
function testConfiguration()
    test_input = LinRange(-100f0, 100f0, 99)

    try
        for left in test_input
            for right in test_input
                for binop in binops
                    test_output = binop.(left, right)
                end
            end
            for unaop in unaops
                test_output = unaop.(left)
            end
        end
    catch error
        @printf("\n\nYour configuration is invalid - one of your operators is not well-defined over the real line.\n\n\n")
        throw(error)
    end
end


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

    for i=1:npopulations
        future = @spawnat :any Population(npop, 3)
        push!(allPops, future)
    end

    # # 2. Start the cycle on every process:
    @sync for i=1:npopulations
        @async allPops[i] = @spawnat :any run(fetch(allPops[i]), ncyclesperiteration, verbosity=verbosity)
    end
    println("Started!")
    cycles_complete = npopulations * niterations

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
                        tmp_pop = run(cur_pop, ncyclesperiteration, verbosity=verbosity)
                        @inbounds @simd for j=1:tmp_pop.n
                            if rand() < 0.1
                                tmp_pop.members[j].tree = simplifyTree(tmp_pop.members[j].tree)
                                tmp_pop.members[j].tree = combineOperators(tmp_pop.members[j].tree)
                                if shouldOptimizeConstants
                                    tmp_pop.members[j] = optimizeConstants(tmp_pop.members[j])
                                end
                            end
                        end
                        if shouldOptimizeConstants
                            #pass #(We already calculate full scores in the optimizer)
                        else
                            tmp_pop = finalizeScores(tmp_pop)
                        end
                        tmp_pop
                    end
                    put!(channels[i], fetch(allPops[i]))
                end

                cycles_complete -= 1
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
            @printf("\n")
            @printf("Cycles per second: %.3e\n", round(average_speed, sigdigits=3))
            @printf("Hall of Fame:\n")
            @printf("-----------------------------------------\n")
            @printf("%-10s  %-8s   %-8s  %-8s\n", "Complexity", "MSE", "Score", "Equation")
            curMSE = baselineMSE
            @printf("%-10d  %-8.3e  %-8.3e  %-.f\n", 0, curMSE, 0f0, avgy)
            lastMSE = curMSE
            lastComplexity = 0

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
                        @printf("%-10d  %-8.3e  %-8.3e  %-s\n" , size, curMSE, score, stringTree(member.tree))
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
