include("hyperparams.jl")
include("dataset.jl")

const maxdegree = 2
const actualMaxsize = maxsize + maxdegree

id = (x,) -> x
const nuna = size(unaops)[1]
const nbin = size(binops)[1]
const nops = nuna + nbin
const nvar = size(X)[2];

function debug(verbosity, string...)
    verbosity > 0 ? println(string...) : nothing
end

# Define a serialization format for the symbolic equations:
mutable struct Node
    #Holds operators, variables, constants in a tree
    degree::Integer #0 for constant/variable, 1 for cos/sin, 2 for +/* etc.
    val::Union{Float32, Integer} #Either const value, or enumerates variable
    constant::Bool #false if variable
    op::Function #enumerates operator (for degree=1,2)
    l::Union{Node, Nothing}
    r::Union{Node, Nothing}

    Node(val::Float32) = new(0, val, true, id, nothing, nothing)
    Node(val::Integer) = new(0, val, false, id, nothing, nothing)
    Node(op, l::Node) = new(1, 0.0f0, false, op, l, nothing)
    Node(op, l::Union{Float32, Integer}) = new(1, 0.0f0, false, op, Node(l), nothing)
    Node(op, l::Node, r::Node) = new(2, 0.0f0, false, op, l, r)

    #Allow to pass the leaf value without additional node call:
    Node(op, l::Union{Float32, Integer}, r::Node) = new(2, 0.0f0, false, op, Node(l), r)
    Node(op, l::Node, r::Union{Float32, Integer}) = new(2, 0.0f0, false, op, l, Node(r))
    Node(op, l::Union{Float32, Integer}, r::Union{Float32, Integer}) = new(2, 0.0f0, false, op, Node(l), Node(r))
end

# Evaluate a symbolic equation:
function evalTree(tree::Node, x::Array{Float32, 1}=Float32[])::Float32
    if tree.degree == 0
        if tree.constant
            return tree.val
        else
            return x[tree.val]
        end
    elseif tree.degree == 1
        return tree.op(evalTree(tree.l, x))
    else
        return tree.op(evalTree(tree.l, x), evalTree(tree.r, x))
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

# Convert an equation to a string
function stringTree(tree::Node)::String
    if tree.degree == 0
        if tree.constant
            return string(tree.val)
        else
            return "x$(tree.val)"
        end
    elseif tree.degree == 1
        return "$(tree.op)($(stringTree(tree.l)))"
    else
        return "$(tree.op)($(stringTree(tree.l)), $(stringTree(tree.r)))"
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
        node.op = unaops[rand(1:length(unaops))]
    else
        node.op = binops[rand(1:length(binops))]
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
    maxChange = T + 1.0f0 + bottom
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
function evalTreeArray(
        tree::Node,
        x::Array{Float32, 2})::Array{Float32, 1}
    len = size(x)[1]
    if tree.degree == 0
        if tree.constant
            return ones(Float32, len) .* tree.val
        else
            return ones(Float32, len) .* x[:, tree.val]
        end
    elseif tree.degree == 1
        return tree.op.(evalTreeArray(tree.l, x))
    else
        return tree.op.(evalTreeArray(tree.l, x), evalTreeArray(tree.r, x))
    end
end

# Sum of square error between two arrays
function SSE(x::Array{Float32}, y::Array{Float32})::Float32
    return sum(((cx,)->cx^2).(x - y))
end

# Mean of square error between two arrays
function MSE(x::Array{Float32}, y::Array{Float32})::Float32
    return SSE(x, y)/size(x)[1]
end

# Score an equation
function scoreFunc(
        tree::Node,
        X::Array{Float32, 2},
        y::Array{Float32, 1};
        parsimony::Float32=0.1f0)::Float32
    try
        return MSE(evalTreeArray(tree, X), y) + countNodes(tree)*parsimony
    catch error
        if isa(error, DomainError)
            return 1f9
        else
            throw(error)
        end
    end
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
            binops[rand(1:length(binops))],
            left,
            right
        )
    else
        newnode = Node(
            unaops[rand(1:length(unaops))],
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

# Select a random node, and replace it an the subtree
# with a variable or constant
function deleteRandomOp(tree::Node)::Node
    node = randomNode(tree)
    # Can "delete" variable or constant too
    if rand() > 0.5
        val = Float32(randn())
    else
        val = rand(1:nvar)
    end
    newnode = Node(val)
    node.l = newnode.l
    node.r = newnode.r
    node.op = newnode.op
    node.degree = newnode.degree
    node.val = newnode.val
    node.constant = newnode.constant
    return tree
end


# Simplify tree 
function simplifyTree(tree::Node)::Node
    if tree.degree == 1
        tree.l = simplifyTree(tree.l)
        if tree.l.degree == 0 && tree.l.constant
            return Node(tree.op(tree.l.val))
        end
    elseif tree.degree == 2
        tree.r = simplifyTree(tree.r)
        tree.l = simplifyTree(tree.l)
        constantsBelow = (
             tree.l.degree == 0 && tree.l.constant &&
             tree.r.degree == 0 && tree.r.constant
        )
        if constantsBelow
            return Node(tree.op(tree.l.val, tree.r.val))
        end
    end
    return tree
end

# Go through one simulated annealing mutation cycle
#  exp(-delta/T) defines probability of accepting a change
function iterate(
        tree::Node, T::Float32,
        X::Array{Float32, 2}, y::Array{Float32, 1},
        alpha::Float32=1.0f0,
        mult::Float32=0.1f0;
        annealing::Bool=true
    )::Node
    prev = tree
    tree = deepcopy(tree)

    mutationChoice = rand()
    weight_for_constant = min(8, countConstants(tree))
    weights = [weight_for_constant, 1, 1, 1, 0.1, 2] .* 1.0
    weights /= sum(weights)
    cweights = cumsum(weights)
    n = countNodes(tree)

    if mutationChoice < cweights[1]
        tree = mutateConstant(tree, T)
    elseif mutationChoice < cweights[2]
        tree = mutateOperator(tree)
    elseif mutationChoice < cweights[3] && n < maxsize
        tree = appendRandomOp(tree)
    elseif mutationChoice < cweights[4]
        tree = deleteRandomOp(tree)
    elseif mutationChoice < cweights[5]
        tree = simplifyTree(tree) # Sometimes we simplify tree
        return tree
    else
        return tree
    end

    if annealing
        beforeLoss = scoreFunc(prev, X, y, parsimony=mult)
        afterLoss = scoreFunc(tree, X, y, parsimony=mult)
        delta = afterLoss - beforeLoss
        probChange = exp(-delta/(T*alpha))

        if isnan(afterLoss) || probChange < rand()
            return deepcopy(prev)
        end
    end

    return tree
end

# Create a random equation by appending random operators
function genRandomTree(length::Integer)::Node
    tree = Node(1.0f0)
    for i=1:length
        tree = appendRandomOp(tree)
    end
    return tree
end


# Define a member of population by equation, score, and age
mutable struct PopMember
    tree::Node
    score::Float32
    birth::Int32

    PopMember(t::Node) = new(t, scoreFunc(t, X, y, parsimony=parsimony), round(Int32, 1e3*(time()-1.6e9)))
    PopMember(t::Node, score::Float32) = new(t, score, round(Int32, 1e3*(time()-1.6e9)))

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

# Return best 10 examples
function bestSubPop(pop::Population)::Population
    best_idx = sortperm([pop.members[member].score for member=1:pop.n])
    return Population(pop.members[best_idx[1:10]])
end

# Mutate the best sampled member of the population
function iterateSample(
        pop::Population, T::Float32;
        annealing::Bool=true)::PopMember
    allstar = bestOfSample(pop)
    new = iterate(
        allstar.tree, T, X, y,
        alpha, parsimony, annealing=annealing)
    allstar.tree = new
    allstar.score = scoreFunc(new, X, y, parsimony=parsimony)
    allstar.birth = round(Int32, 1e3*(time()-1.6e9))
    return allstar
end

# Pass through the population several times, replacing the oldest
# with the fittest of a small subsample
function regEvolCycle(
    pop::Population, T::Float32;
    annealing::Bool=true)::Population
    for i=1:Integer(pop.n/ns)
        baby = iterateSample(pop, T, annealing=annealing)
        #printTree(baby.tree)
        oldest = argmin([pop.members[member].birth for member=1:pop.n])
        pop.members[oldest] = baby
    end
    return pop
end

# Cycle through regularized evolution many times,
# printing the fittest equation every 10% through
function run(
        pop::Population,
        ncycles::Integer,
        annealing::Bool=false;
        verbosity::Integer=0
        )::Population

    allT = LinRange(1.0f0, 0.0f0, ncycles)
    for iT in 1:size(allT)[1]
        if annealing
            pop = regEvolCycle(pop, allT[iT], annealing=true)
        else
            pop = regEvolCycle(pop, 1.0f0, annealing=true)
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
