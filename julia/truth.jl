# *** Custom Functions
##################################################################################################################################
# *** Will somewhere need to define a list TRUTHS of all valid auxliary truths
struct Transformation
    type::Integer # 1 is symmetry, 2 is zero, 3 is equality
    params::Array{Int32}
    Transformation(type::Integer, params::Array{Int32}) = new(type, params)
    Transformation(type::Integer, params::Array{Int64}) = new(type, params)

end
struct Truth
    transformation::Transformation
    weights::Array{Float32}
    Truth(transformation::Transformation, weights::Array{Float32}) = new(transformation, weights)
    Truth(type::Int64, params::Array{Int64}, weights::Array{Float32}) = new(Transformation(type, params), weights)
    Truth(transformation::Transformation, weights::Array{Float64}) = new(transformation, weights)
    Truth(type::Int64, params::Array{Int64}, weights::Array{Float64}) = new(Transformation(type, params), weights)
end
# Returns a linear combination when given X of shape nxd, y of shape nx1 is f(x) and w of shape d+2x1, result is shape nx1
function LinearPrediction(cX::Array{Float32}, cy::Array{Float32}, w::Array{Float32})::Array{Float32}
     preds = 0
     for i in 1:ndims(cX)
       preds = preds .+ cX[:,i].*w[i]
       end
     preds = preds .+ cy.*w[ndims(cX)+1]
     return preds .+ w[ndims(cX)+2]
end

# Returns a copy of the data with the two specified columns swapped
function swapColumns(cX::Array{Float32, 2}, a::Integer, b::Integer)::Array{Float32, 2}
    X1 = copy(cX)
    X1[:, a] = cX[:, b]
    X1[:, b] = cX[:, a]
    return X1
end

# Returns a copy of the data with the specified integers in the list set to value given
function setVal(cX::Array{Float32, 2}, a::Array{Int32, 1}, val::Float32)::Array{Float32, 2}
    X1 = copy(cX)
    for i in 1:size(a)[1]
        X1[:, a[i]] = fill!(cX[:, a[i]], val)
    end
    return X1
end

# Returns a copy of the data with the specified integer indices in the list set to the first item of that list
function setEq(cX::Array{Float32, 2}, a::Array{Int32, 1})::Array{Float32, 2}
    X1 = copy(cX)
    val = X1[:, a[1]]
    for i in 1:size(a)[1]
        X1[:, a[i]] = val
    end
    return X1
end

# Takes in a dataset and returns the transformed version of it as per the specified type and parameters
function transform(cX::Array{Float32, 2}, transformation::Transformation)::Array{Float32, 2}
    if transformation.type==1 # then symmetry
        a = transformation.params[1]
        b = transformation.params[2]
        return swapColumns(cX, a, b)
    elseif transformation.type==2 # then zero condition
        return setVal(cX, transformation.params, Float32(0))
    elseif transformation.type == 3 # then equality condition
        return setEq(cX, transformation.params)
    else # Then error return X
        return cX
    end
end
function transform(cX::Array{Float32, 2}, truth::Truth)::Array{Float32, 2}
    return transform(cX, truth.transformation)
end

# Takes in X that has been transformed and returns what the Truth projects the target values should be
function truthPrediction(X_transformed::Array{Float32, 2}, cy::Array{Float32}, truth::Truth)::Array{Float32}
    return LinearPrediction(X_transformed, cy, truth.weights)
end
