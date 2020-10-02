
#Store each binary tree as an array.
#Full population is 2D array.
mutable struct Population
    exists::Array{Bool, 2}
	degree::Array{Integer, 2}
	val::Array{Float32, 2}
	idx::Array{Integer, 2}
	constant::Array{Bool, 2}
	op::Array{Integer, 2}

    Population(depth::Integer, n::Integer) = new(
            zeros(Bool, (n, 2^depth)),
            zeros(Integer, (n, 2^depth)),
            zeros(Float32, (n, 2^depth)),
            zeros(Integer, (n, 2^depth)),
            zeros(Bool, (n, 2^depth)),
            zeros(Integer, (n, 2^depth))
        )
end

Tree = Tuple{Population, Int}
Node = Tuple{Population, Int, Int}

function makeRandomLeaf(node::Node)::Population
    pop, t, i = node
    pop.exists[t, i] = true
    pop.degree[t, i] = 0
    make_constant = rand() < 0.5
    if 
        pop.constant[t, i] = false
        pop.idx[t, i] = rand(1:nvar)
    else
        pop.constant[t, i] = true
        pop.val[t, i] = Float32(randn())
    end
end

#Turn into for loop instead.
function getRandomLeaf(node::Node)::Node
    pop, t, i = node
    degree = pop.degree[t, i]
    left_child = 2*i + 1
    right_child = 2*i + 2

    if degree == 0
        return node
    elseif degree == 1
        left_child = 2*i + 1
        return getRandomLeaf((pop, t, left_child))
    else
        if rand() < 0.5
            return getRandomLeaf((pop, t, left_child))
        else
            return getRandomLeaf((pop, t, right_child))
        end
    end
end

function appendNode(tree::Tree)::Population
    pop, t = tree

    n_nodes = sum(pop.exists[t, :])
    if n_nodes == 0 #Empty tree
        makeRandomLeaf((pop, t, 1))
    else
        leaf = getRandomLeaf()
        
    end
end
