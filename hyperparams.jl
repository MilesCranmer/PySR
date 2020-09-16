# Define allowed operators
plus(x::Float32, y::Float32)::Float32 = x+y
mult(x::Float32, y::Float32)::Float32 = x*y;

##########################
# # Allowed operators
# (Apparently using const for globals helps speed)
const binops = [plus, mult]
const unaops = [sin, cos, exp]
##########################

# How many equations to search when replacing
const ns=10;

##################
# Hyperparameters
# How much to punish complexity
const parsimony = 1f-3
# How much to scale temperature by (T between 0 and 1)
const alpha = 10.0f0
# Max size of an equation (too large will slow program down)
const maxsize = 20
# Whether to migrate between threads (you should)
const migration = true
# Whether to re-introduce best examples seen (helps a lot)
const hofMigration = true
# Fraction of population to replace with hall of fame
const fractionReplacedHof = 0.1f0
# Optimize constants
const shouldOptimizeConstants = true
##################


