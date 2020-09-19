# Define allowed operators. Any julia operator can also be used.
plus(x::Float32, y::Float32)::Float32 = x+y #Do not change the name of this operator.
mult(x::Float32, y::Float32)::Float32 = x*y #Do not change the name of this operator.
pow(x::Float32, y::Float32)::Float32 = sign(x)*abs(x)^y
div(x::Float32, y::Float32)::Float32 = x/y
loga(x::Float32)::Float32 = log(abs(x) + 1)
