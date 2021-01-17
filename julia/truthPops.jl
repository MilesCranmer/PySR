# Returns the MSE between the predictions and the truth provided targets for the given dataset
function truthScore(member::PopMember, cX::Array{Float32, 2}, cy::Array{Float32}, truth::Truth)::Float32
    transformed = transform(cX, truth)
    targets = truthPrediction(transformed, cy, truth)
    preds = evalTreeArray(member.tree, transformed)
    return MSE(preds, targets)
end

# Assumes a dataset X, y for a given truth
function truthScore(member::PopMember, truth::Truth)::Float32
    return truthScore(member, X, y, truth)
end

# Assumes a list of Truths TRUTHS is defined. Performs the truthScore function for each of them and returns the average
function truthScore(member::PopMember, cX::Array{Float32, 2}, cy::Array{Float32})::Float32
    s = 0
    for truth in TRUTHS
        s += (truthScore(member, cX, cy, truth))/size(TRUTHS)[1]
    end
    return s
end

# Assumes list of Truths TRUTHS and dataset X, y are defined
function truthScore(member::PopMember)::Float32
    return truthScore(member, X, y)
end
# Returns the MSE between the predictions and the truth provided targets for the given dataset
function truthScore(tree::Node, cX::Array{Float32, 2}, cy::Array{Float32}, truth::Truth)::Float32
    transformed = transform(cX, truth)
    targets = truthPrediction(transformed, cy, truth)
    preds = evalTreeArray(tree, transformed)
    return MSE(preds, targets)
end

# Assumes a dataset X, y for a given truth
function truthScore(tree::Node, truth::Truth)::Float32
    return truthScore(tree, X, y, truth)
end

# Assumes a list of Truths TRUTHS is defined. Performs the truthScore function for each of them and returns the average
function truthScore(tree::Node, cX::Array{Float32, 2}, cy::Array{Float32})::Float32
    s = 0
    for truth in TRUTHS
        s += (truthScore(tree, cX, cy, truth))/size(TRUTHS)[1]
    end
    return s
end

# Assumes list of Truths TRUTHS and dataset X, y are defined
function truthScore(tree::Node)::Float32
    return truthScore(tree, X, y)
end

# Returns true iff Truth Score is below a given threshold i.e truth is satisfied
function testTruth(member::PopMember, truth::Truth, threshold::Float32=Float32(1.0e-8))::Bool
    truthError = truthScore(member, truth)
    #print(stringTree(member.tree), "\n")
    #print(truth, ": ")
    #print(truthError, "\n")
    if truthError > threshold
        #print("Returns False \n ----\n")
        return false
    else
        #print("Returns True \n ----\n")
        return true
    end
end

# Returns a list of violating functions from assumed list TRUTHS
function violatingTruths(member::PopMember)::Array{Truth}
    return violatingTruths(member.tree)
end

# Returns true iff Truth Score is below a given threshold i.e truth is satisfied
function testTruth(tree::Node, truth::Truth, threshold::Float32=Float32(1.0e-3))::Bool
    truthError = truthScore(tree, truth)
    if truthError > threshold
        return false
    else
        return true
    end
end

# Returns a list of violating functions from assumed list TRUTHS
function violatingTruths(tree::Node)::Array{Truth}
    toReturn = []
    #print("\n Checking Equation ", stringTree(tree), "\n")
    for truth in TRUTHS
        test_truth = testTruth(tree, truth)
        #print("Truth: ", truth, ": " , test_truth, "\n-----\n")
        if !test_truth
            append!(toReturn, [truth])
        end
    end
    return toReturn
end

function randomIndex(cX::Array{Float32, 2}, k::Integer=10)::Array{Int32, 1}
    indxs = sample([Int32(i) for i in 1:size(cX)[1]], k)
    return indxs
end

function randomIndex(leng::Integer, k::Integer=10)::Array{Int32, 1}
    indxs = sample([Int32(i) for i in 1:leng], k)
    return indxs
end

function extendedX(cX::Array{Float32, 2}, truth::Truth, indx::Array{Int32, 1})::Array{Float32, 2}
    workingcX = copy(cX)
    X_slice = workingcX[indx, :]
    X_transformed = transform(X_slice, truth)
    return X_transformed
end
function extendedX(truth::Truth, indx::Array{Int32, 1})::Union{Array{Float32, 2}, Nothing}
    return extendedX(OGX, truth, indx)
end
function extendedX(cX::Array{Float32, 2}, violatedTruths::Array{Truth}, indx::Array{Int32, 1})::Union{Array{Float32, 2}, Nothing}
    if length(violatedTruths) == 0
        return nothing
    end
    workingX = extendedX(cX, violatedTruths[1], indx)
    for truth in violatedTruths[2:length(violatedTruths)]
        workingX = vcat(workingX, extendedX(cX, truth, indx))
    end
    return workingX
end
function extendedX(violatedTruths::Array{Truth}, indx::Array{Int32, 1})::Union{Array{Float32, 2}, Nothing}
    return extendedX(OGX, violatedTruths, indx)
end
function extendedX(tree::Node, indx::Array{Int32, 1})::Union{Array{Float32, 2}, Nothing}
    violatedTruths = violatingTruths(tree)
    return extendedX(violatedTruths, indx)
end
function extendedX(member::PopMember, indx::Array{Int32, 1})::Union{Array{Float32, 2}, Nothing}
    return extendedX(member.tree, indx)
end


function extendedy(cX::Array{Float32, 2}, cy::Array{Float32}, truth::Truth, indx::Array{Int32, 1})::Union{Array{Float32}, Nothing}
    cy = copy(cy)
    cX = copy(cX)
    X_slice = cX[indx, :]
    y_slice = cy[indx]
    X_transformed = transform(X_slice, truth)
    y_transformed = truthPrediction(X_transformed, y_slice, truth)
    return y_transformed
end
function extendedy(truth::Truth, indx::Array{Int32, 1})::Union{Array{Float32}, Nothing}
    return extendedy(OGX, OGy, truth, indx)
end
function extendedy(cX::Array{Float32, 2}, cy::Array{Float32}, violatedTruths::Array{Truth}, indx::Array{Int32, 1})::Union{Array{Float32}, Nothing}
    if length(violatedTruths) == 0
        return nothing
    end
    workingy = extendedy(cX, cy, violatedTruths[1], indx)
    for truth in violatedTruths[2:length(violatedTruths)]
        workingy = vcat(workingy, extendedy(cX, cy, truth, indx))
    end
    return workingy
end
function extendedy(violatedTruths::Array{Truth}, indx::Array{Int32, 1})::Union{Array{Float32}, Nothing}
    return extendedy(OGX,OGy, violatedTruths, indx)
end
function extendedy(tree::Node, indx::Array{Int32, 1})::Union{Array{Float32}, Nothing}
    violatedTruths = violatingTruths(tree)
    return extendedy(violatedTruths, indx)
end
function extendedy(member::PopMember, indx::Array{Int32, 1})::Union{Array{Float32}, Nothing}
    return extendedy(member.tree, indx)
end