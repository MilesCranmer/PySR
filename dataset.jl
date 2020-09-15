# Here is the function we want to learn (x2^2 + cos(x3) + 5)

##########################
# # Dataset to learn
const X = convert(Array{Float32, 2}, randn(100, 5)*2)
const y = convert(Array{Float32, 1}, ((cx,)->cx^2).(X[:, 2]) + cos.(X[:, 3]) .- 5)
##########################
