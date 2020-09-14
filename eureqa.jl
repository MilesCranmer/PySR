using ProgressBars

# Define allowed operators
plus(x::Float64, y::Float64) = x+y
mult(x::Float64, y::Float64) = x*y;

# (Apparently using const for globals helps speed)
const binops = [plus, mult]
const unaops = [sin, cos, exp];

const nvar = 5;
const X = rand(100, nvar);

# Here is the function we want to learn (x2^2 + cos(x3) + 5)
const y = ((cx,)->cx^2).(X[:, 2]) + cos.(X[:, 3]) .+ 5.0;

# How much to punish complexity
const parsimony = 0.01
# How much to scale temperature by (T between 0 and 1)
const alpha = 10.0




id = (x,) -> x
const nuna = size(unaops)[1]
const nbin = size(binops)[1]
const nops = nuna + nbin

# Define a serialization format for the symbolic equations:
mutable struct Node
    #Holds operators, variables, constants in a tree
    degree::Int #0 for constant/variable, 1 for cos/sin, 2 for +/* etc.
    val::Union{Float64, Int} #Either const value, or enumerates variable
    constant::Bool #false if variable 
    op::Function #enumerates operator (for degree=1,2)
    l::Union{Node, Nothing}
    r::Union{Node, Nothing}
    
    Node(val::Float64) = new(0, val, true, id, nothing, nothing)
    Node(val::Int) = new(0, val, false, id, nothing, nothing)
    Node(op, l::Node) = new(1, 0.0, false, op, l, nothing)
    Node(op, l::Union{Float64, Int}) = new(1, 0.0, false, op, Node(l), nothing)
    Node(op, l::Node, r::Node) = new(2, 0.0, false, op, l, r)
    
    #Allow to pass the leaf value without additional node call:
    Node(op, l::Union{Float64, Int}, r::Node) = new(2, 0.0, false, op, Node(l), r)
    Node(op, l::Node, r::Union{Float64, Int}) = new(2, 0.0, false, op, l, Node(r))
    Node(op, l::Union{Float64, Int}, r::Union{Float64, Int}) = new(2, 0.0, false, op, Node(l), Node(r))
end

# Evaluate a symbolic equation:
function evalTree(tree::Node, x::Array{Float64, 1}=Float64[])::Float64
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
function countNodes(tree::Node)::Int
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
function countUnaryOperators(tree::Node)::Int
    if tree.degree == 0
        return 0
    elseif tree.degree == 1
        return 1 + countUnaryOperators(tree.l)
    else
        return 0 + countUnaryOperators(tree.l) + countUnaryOperators(tree.r)
    end
end

# Count the number of binary operators in the equation
function countBinaryOperators(tree::Node)::Int
    if tree.degree == 0
        return 0
    elseif tree.degree == 1
        return 0 + countBinaryOperators(tree.l)
    else
        return 1 + countBinaryOperators(tree.l) + countBinaryOperators(tree.r)
    end
end

# Count the number of operators in the equation
function countOperators(tree::Node)::Int
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
function countConstants(tree::Node)::Int
    if tree.degree == 0
        return convert(Int, tree.constant)
    elseif tree.degree == 1
        return 0 + countConstants(tree.l)
    else
        return 0 + countConstants(tree.l) + countConstants(tree.r)
    end
end

# Randomly perturb a constant
function mutateConstant(
        tree::Node, T::Float64,
        probNegate::Float64=0.01)::Node
    # T is between 0 and 1.
    
    if countConstants(tree) == 0
        return tree
    end
    node = randomNode(tree)
    while node.degree != 0 || node.constant == false
        node = randomNode(tree)
    end
    
    maxChange = T + 1.0
    factor = maxChange^rand()
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
        x::Array{Float64, 2})::Array{Float64, 1}
    return mapslices(
        (cx,) -> evalTree(tree, cx),
        x,
        dims=[2]
    )[:, 1]
end

# Sum of square error between two arrays
function SSE(x::Array{Float64}, y::Array{Float64})::Float64
    return sum(((cx,)->cx^2).(x - y))
end

# Mean of square error between two arrays
function MSE(x::Array{Float64}, y::Array{Float64})::Float64
    return SSE(x, y)/size(x)[1]
end

# Score an equation
function scoreFunc(
        tree::Node,
        X::Array{Float64, 2},
        y::Array{Float64, 1},
        parsimony::Float64=0.1)::Float64
    return MSE(evalTreeArray(tree, X), y) + countNodes(tree)*parsimony
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
        left = randn()
    else
        left = rand(1:nvar)
    end
    if rand() > 0.5
        right = randn()
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
        val = randn()
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

# Go through one simulated annealing mutation cycle
#  exp(-delta/T) defines probability of accepting a change
function iterate(
        tree::Node, T::Float64,
        X::Array{Float64, 2}, y::Array{Float64, 1},
        alpha::Float64=1.0,
        mult::Float64=0.1
    )::Node
    prev = deepcopy(tree)
    
    mutationChoice = rand()
    weights = [8, 1, 1, 1]
    weights /= sum(weights)
    cweights = cumsum(weights)
    
    if mutationChoice < cweights[1]
        tree = mutateConstant(tree, T)
    elseif mutationChoice < cweights[2]
        tree = mutateOperator(tree)
    elseif mutationChoice < cweights[3]
        tree = appendRandomOp(tree)
    elseif mutationChoice < cweights[4]
        tree = deleteRandomOp(tree)
    end
    
    try
        beforeLoss = scoreFunc(prev, X, y, mult)
        afterLoss = scoreFunc(tree, X, y, mult)
        delta = afterLoss - beforeLoss
        probChange = exp(-delta/(T*alpha))

        if probChange > rand()
            return tree
        end

        return prev
    catch error
        # Sometimes too many chained exp operators
        if isa(error, DomainError)
            return prev
        else
            throw(error)
        end
    end
end

# Create a random equation by appending random operators
function genRandomTree(length::Int)::Node
    tree = Node(1.0)
    for i=1:length
        tree = appendRandomOp(tree)
    end
    return tree
end


# Define a member of population by equation, score, and age
mutable struct PopMember
    tree::Node
    score::Float64
    birth::Float64
    
    PopMember(t) = new(t, scoreFunc(t, X, y, parsimony), time()-1.6e9)
end

# A list of members of the population, with easy constructors,
#  which allow for random generation of new populations
mutable struct Population
    members::Array{PopMember, 1}
    n::Int
    
    Population(pop::Array{PopMember, 1}) = new(pop, size(pop)[1])
    Population(npop::Int64) = new([PopMember(genRandomTree(3)) for i=1:npop], npop)
    Population(npop::Int64, nlength::Int64) = new([PopMember(genRandomTree(nlength)) for i=1:npop], npop)
    
end

# Sample 10 random members of the population, and make a new one
function samplePop(pop::Population)::Population
    idx = rand(1:pop.n, 10)
    return Population(pop.members[idx])#Population(deepcopy(pop.members[idx]))
end

# Sample the population, and get the best member from that sample
function bestOfSample(pop::Population)::PopMember
    sample = samplePop(pop)
    best_idx = argmin([sample.members[member].score for member=1:sample.n])
    return sample.members[best_idx]
end

# Mutate the best sampled member of the population
function iterateSample(pop::Population, T::Float64)::PopMember
    allstar = bestOfSample(pop)
    new = iterate(allstar.tree, T, X, y, alpha, parsimony)
    allstar.tree = new
    allstar.score = scoreFunc(new, X, y, parsimony)
    allstar.birth = time() - 1.6e9
    return allstar
end

# Pass through the population several times, replacing the oldest
# with the fittest of a small subsample
function regEvolCycle(pop::Population, T::Float64)::Population
    for i=1:Int(pop.n/10)
        baby = iterateSample(pop, T)
        oldest = argmin([pop.members[member].birth for member=1:pop.n])
        pop.members[oldest] = baby
    end
    return pop
end

# Cycle through regularized evolution many times,
# printing the fittest equation every 10% through
function run(ncycles::Int,
        npop::Int=100,
        annealing::Bool=false)::Population

    allT = LinRange(1.0, 0.0, ncycles)
    pop = Population(npop, 3)
    bestScore = Inf
    for iT in tqdm(1:size(allT)[1])
        if annealing
            pop = regEvolCycle(pop, allT[iT])
        else
            pop = regEvolCycle(pop, 0.0)
        end
        bestCurScoreIdx = argmin([pop.members[member].score for member=1:pop.n])
        bestCurScore = pop.members[bestCurScoreIdx].score
        if bestCurScore < bestScore
            bestScore = bestCurScore
            println(bestScore, " is the score for ", stringTree(pop.members[bestCurScoreIdx].tree))
        end
    end
    return pop
end

println("Lets try to learn (x2^2 + cos(x3) + 5) using regularized evolution from scratch")
pop = run(10000, 1000, false);


