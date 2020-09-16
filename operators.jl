# Define allowed operators. Any julia operator can also be used.
plus(x::Float32, y::Float32)::Float32 = x+y
mult(x::Float32, y::Float32)::Float32 = x*y;
pow(x::Float32, y::Float32)::Float32 = sign(x)*abs(x)^y;
