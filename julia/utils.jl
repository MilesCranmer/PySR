function id(x::Float32)::Float32
    x
end

function debug(verbosity, string...)
    verbosity > 0 ? println(string...) : nothing
end

function getTime()::Integer
    return round(Integer, 1e3*(time()-1.6e9))
end