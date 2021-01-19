# Define a member of population by equation, score, and age
mutable struct PopMember
    tree::Node
    score::Float32
    birth::Integer

    PopMember(t::Node) = new(t, scoreFunc(t), getTime())
    PopMember(t::Node, score::Float32) = new(t, score, getTime())

end