# A list of members of the population, with easy constructors,
#  which allow for random generation of new populations
mutable struct Population
    members::Array{PopMember, 1}
    n::Integer

    Population(pop::Array{PopMember, 1}) = new(pop, size(pop)[1])
    Population(npop::Integer) = new([PopMember(genRandomTree(3)) for i=1:npop], npop)
    Population(npop::Integer, nlength::Integer) = new([PopMember(genRandomTree(nlength)) for i=1:npop], npop)

end

# Sample 10 random members of the population, and make a new one
function samplePop(pop::Population)::Population
    idx = rand(1:pop.n, ns)
    return Population(pop.members[idx])
end

# Sample the population, and get the best member from that sample
function bestOfSample(pop::Population)::PopMember
    sample = samplePop(pop)
    best_idx = argmin([sample.members[member].score for member=1:sample.n])
    return sample.members[best_idx]
end

function finalizeScores(pop::Population)::Population
    need_recalculate = batching
    if need_recalculate
        @inbounds @simd for member=1:pop.n
            pop.members[member].score = scoreFunc(pop.members[member].tree)
        end
    end
    return pop
end

# Return best 10 examples
function bestSubPop(pop::Population; topn::Integer=10)::Population
    best_idx = sortperm([pop.members[member].score for member=1:pop.n])
    return Population(pop.members[best_idx[1:topn]])
end