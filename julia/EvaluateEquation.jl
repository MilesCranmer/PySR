# Evaluate an equation over an array of datapoints
function evalTreeArray(tree::Node)::Union{Array{Float32, 1}, Nothing}
    return evalTreeArray(tree, X)
end


# Evaluate an equation over an array of datapoints
function evalTreeArray(tree::Node, cX::Array{Float32, 2})::Union{Array{Float32, 1}, Nothing}
    clen = size(cX)[1]
    if tree.degree == 0
        if tree.constant
            return fill(tree.val, clen)
        else
            return copy(cX[:, tree.val])
        end
    elseif tree.degree == 1
        cumulator = evalTreeArray(tree.l, cX)
        if cumulator === nothing
            return nothing
        end
        op_idx = tree.op
        UNAOP!(cumulator, op_idx, clen)
        @inbounds for i=1:clen
            if isinf(cumulator[i]) || isnan(cumulator[i])
                return nothing
            end
        end
        return cumulator
    else
        cumulator = evalTreeArray(tree.l, cX)
        if cumulator === nothing
            return nothing
        end
        array2 = evalTreeArray(tree.r, cX)
        if array2 === nothing
            return nothing
        end
        op_idx = tree.op
        BINOP!(cumulator, array2, op_idx, clen)
        @inbounds for i=1:clen
            if isinf(cumulator[i]) || isnan(cumulator[i])
                return nothing
            end
        end
        return cumulator
    end
end
