# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export MVector

# ~~~ INTERFACE FOR MVector and MMatrix ~~~
# The type parameter `X` must support
# 1) dot(::X, ::X)
# 2) similar(::X)
# 3) full broadcast functionality, with variables of type `X` and scalars

# ~~~ Vector Type ~~~
mutable struct MVector{X, N, NS}
    x::NTuple{N, X}        # the seed along the orbit
    d::NTuple{NS, Float64} # period and optional NS-1 shifts
    MVector(x::NTuple{N, X}, T::Real, s::Real) where {N, X} =
        new{X, N}(x, Float64.((T, s)))
    MVector(x::NTuple{N, X}, T::Real) where {N, X} =
        new{X, N}(x, Float64.((T, )))
end

# getindex to have z[i] mean z.x[i]
Base.getindex(z::MVector, i::Int) = z.x[i]

# interface for GMRES solver
Base.similar(z::MVector) = MVector(similar.(z.x), z.d...)
Base.copy(z::MVector) = MVector(copy.(z.x), z.d...)
Base.norm(z::MVector) = sqrt(dot(z, z))
Base.dot(a::MVector{X, N}, b::MVector{X, N}) where {X, N} =
    sum(a.d.*b.d) + sum(dot.(a.x, b.x))

# define stuff necessary to use . notation with MVector
_get_seed(z::MVector, i::Int) = z.x[i]
_get_d(z::MVector) = z.d
_get_seed(z, i) = z
_get_d(z) = z

@generated function Base.Broadcast.broadcast!(f,
                                              dest::MVector{X, N},
                                              args::Vararg{Any, n}) where {X, N, n}
    expr = quote
        dest.d = broadcast(f, map(_get_d, args)...)
        return dest
    end
    for i = 1:N
        rhs = [:(_get_seed(args[$j], $i)) for j = 1:n]
        unshift!(expr.args, # push front
            :(broadcast!(f, _get_seed(dest, $i), $(rhs...))))
    end
    unshift!(expr.args, Expr(:meta, :inline))
    return expr
end