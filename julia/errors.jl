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