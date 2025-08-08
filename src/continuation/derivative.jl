# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export SolCache

# weights for lagrange interpolation of the derivative of a function
@inline weights(s, s0) = ()
@inline weights(s, s0, s1) = 
    ((s0-s1), (s1-s0))
@inline weights(s, s0, s1, s2) = 
    ((s-s2)/((s0-s1)*(s0-s2)) + (s-s1)/((s0-s2)*(s0-s1)),
     (s-s2)/((s1-s0)*(s1-s2)) + (s-s0)/((s1-s2)*(s1-s0)),
     (s-s1)/((s2-s0)*(s2-s1)) + (s-s0)/((s2-s1)*(s2-s0)))
@inline weights(s, s0, s1, s2, s3) =
    (((s-s1)*(s-s2) + (s-s1)*(s-s3) + (s-s2)*(s-s3))/((s0-s1)*(s0-s2)*(s0-s3)),
     ((s-s0)*(s-s2) + (s-s0)*(s-s3) + (s-s2)*(s-s3))/((s1-s0)*(s1-s2)*(s1-s3)),
     ((s-s1)*(s-s0) + (s-s1)*(s-s3) + (s-s0)*(s-s3))/((s2-s0)*(s2-s1)*(s2-s3)),
     ((s-s1)*(s-s2) + (s-s1)*(s-s0) + (s-s2)*(s-s0))/((s3-s0)*(s3-s1)*(s3-s2)))

# object to store past solution at different values of the arclength
struct SolCache{X}
    xs::Vector{X}
    ss::Vector{Float64}
    SolCache(x::X, s::Real) where {X} = new{X}(X[copy(x)], Float64[s])
end

# push solution and arc length
function Base.push!(sc::SolCache{X}, x::X, s::Real) where {X}
    # check we are increasing on s
    s > sc.ss[end] || throw(ArgumentError("arclength should be increasing"))
    push!(sc.xs, x)
    push!(sc.ss, s)
    while length(sc.xs) > 4
        popfirst!(sc.xs); popfirst!(sc.ss)
    end
    return sc
end

# compute derivative dx/ds at the last point that was pushed 
function dds!(sc::SolCache{X}, out::X) where {X}
    out .= 0
    ws = weights(sc.ss[end], sc.ss...)
    for i = 1:length(ws)
        out .+= ws[i].*sc.xs[i]
    end
    return out
end