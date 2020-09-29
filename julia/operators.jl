import SpecialFunctions: gamma, lgamma, erf, erfc, beta

# Define allowed operators. Any julia operator can also be used.
plus(x::Float32, y::Float32)::Float32 = x+y #Do not change the name of this operator.
mult(x::Float32, y::Float32)::Float32 = x*y #Do not change the name of this operator.
pow(x::Float32, y::Float32)::Float32 = sign(x)*abs(x)^y
div(x::Float32, y::Float32)::Float32 = x/y
logm(x::Float32)::Float32 = log(abs(x) + 1f-8)
logm2(x::Float32)::Float32 = log2(abs(x) + 1f-8)
logm10(x::Float32)::Float32 = log10(abs(x) + 1f-8)
sqrtm(x::Float32)::Float32 = sqrt(abs(x))
neg(x::Float32)::Float32 = -x

function greater(x::Float32, y::Float32)::Float32
    if x > y
        return 1f0
    end
    return 0f0
end

function relu(x::Float32)::Float32
    if x > 0f0
        return x
    end
    return 0f0
end

function logical_or(x::Float32, y::Float32)::Float32
    if x > 0f0 || y > 0f0
        return 1f0
    end
    return 0f0
end

# (Just use multiplication normally)
function logical_and(x::Float32, y::Float32)::Float32
    if x > 0f0 && y > 0f0
        return 1f0
    end
    return 0f0
end
