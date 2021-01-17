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
        if cumulator === nothing
            return nothing
        end
        op_idx = tree.op
        UNAOP!(cumulator, op_idx, clen)
        @inbounds for i=1:clen
            if isinf(cumulator[i]) || isnan(cumulator[i])
                return nothing
            end
        end
        return cumulator
    else
        cumulator = evalTreeArray(tree.l, cX)
        if cumulator === nothing
            return nothing
        end
        array2 = evalTreeArray(tree.r, cX)
        if array2 === nothing
            return nothing
        end
        op_idx = tree.op
        BINOP!(cumulator, array2, op_idx, clen)
        @inbounds for i=1:clen
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
    if prediction === nothing
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
    if prediction === nothing
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