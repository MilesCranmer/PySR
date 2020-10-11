import SpecialFunctions: gamma, lgamma, erf, erfc, beta


import Base.FastMath: sqrt_llvm_fast, neg_float_fast,
    add_float_fast, sub_float_fast, mul_float_fast, div_float_fast, rem_float_fast,
    eq_float_fast, ne_float_fast, lt_float_fast, le_float_fast,
    sign_fast, abs_fast, log_fast, log2_fast, log10_fast, sqrt_fast,
    pow_fast

# Implicitly defined:
#binary: mod
#unary: exp, abs, log1p, sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh, erf, erfc, gamma, relu, round, floor, ceil, round, sign.

# Use some fast operators from https://github.com/JuliaLang/julia/blob/81597635c4ad1e8c2e1c5753fda4ec0e7397543f/base/fastmath.jl
# Define allowed operators. Any julia operator can also be used.
plus(x::Float32, y::Float32)::Float32 = add_float_fast(x, y) #Do not change the name of this operator.
sub(x::Float32, y::Float32)::Float32 = sub_float_fast(x, y) #Do not change the name of this operator.
mult(x::Float32, y::Float32)::Float32 = mul_float_fast(x, y) #Do not change the name of this operator.
pow(x::Float32, y::Float32)::Float32 = sign_fast(x)*pow_fast(abs(x), y)
div(x::Float32, y::Float32)::Float32 = div_float_fast(x, y)
logm(x::Float32)::Float32 = log_fast(abs_fast(x) + 1f-8)
logm2(x::Float32)::Float32 = log2_fast(abs_fast(x) + 1f-8)
logm10(x::Float32)::Float32 = log10_fast(abs_fast(x) + 1f-8)
sqrtm(x::Float32)::Float32 = sqrt_fast(abs_fast(x))
neg(x::Float32)::Float32 = sub_float_fast(x)

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
