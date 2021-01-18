import Random: randperm

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

if weighted
    const avgy = sum(y .* weights)/sum(weights)
    const baselineMSE = MSE(y, convert(Array{Float32, 1}, ones(len) .* avgy), weights)
else
    const avgy = sum(y)/len
    const baselineMSE = MSE(y, convert(Array{Float32, 1}, ones(len) .* avgy))
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
