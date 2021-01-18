# Check if any binary operator are overly complex
function flagBinOperatorComplexity(tree::Node, op::Int)::Bool
    if tree.degree == 0
        return false
    elseif tree.degree == 1
        return flagBinOperatorComplexity(tree.l, op)
    else
        if tree.op == op
            overly_complex = (
                    ((bin_constraints[op][1] > -1) &&
                     (countNodes(tree.l) > bin_constraints[op][1]))
                      ||
                    ((bin_constraints[op][2] > -1) &&
                     (countNodes(tree.r) > bin_constraints[op][2]))
                )
            if overly_complex
                return true
            end
        end
        return (flagBinOperatorComplexity(tree.l, op) || flagBinOperatorComplexity(tree.r, op))
    end
end

# Check if any unary operators are overly complex
function flagUnaOperatorComplexity(tree::Node, op::Int)::Bool
    if tree.degree == 0
        return false
    elseif tree.degree == 1
        if tree.op == op
            overly_complex = (
                      (una_constraints[op] > -1) &&
                      (countNodes(tree.l) > una_constraints[op])
                )
            if overly_complex
                return true
            end
        end
        return flagUnaOperatorComplexity(tree.l, op)
    else
        return (flagUnaOperatorComplexity(tree.l, op) || flagUnaOperatorComplexity(tree.r, op))
    end
end
