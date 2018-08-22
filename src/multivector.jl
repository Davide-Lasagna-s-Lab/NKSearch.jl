# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export MVector

# ~~~ INTERFACE FOR MVector and NKMatrix ~~~
# The type parameter `X` must support
# 1) dot(::X, ::X)
# 2) similar(::X)
# 3) full broadcast functionality, with variables of type `X` and scalars, by
#    overloading `Base.broadcast!(f, ::X, ::Vararg{X, N}) where N

# ~~~ Vector Type ~~~
mutable struct MVector{X, N}
    x::NTuple{N, X}       # the seeds along the orbit
    T::NTuple{3, Float64} # orbit period
    s::Float64            # shift
end

# outer constructor
MVector(x::NTuple{N, X}, T::NTuple{3, Real}, s::Real) where {X, N} = 
    MVector{X, N}(x, convert(NTuple{3, Float64}, T), s)

# getindex to have z[i] mean z.x[i]
Base.getindex(z::MVector, i::Int) = z.x[i]

# interface for GMRES solver
Base.similar(z::MVector) = MVector(similar.(z.x), (0.0, 0.0, 0.0), 0.0)
Base.copy(z::MVector) = MVector(copy.(z.x), z.T, z.s)
Base.norm(z::MVector) = sqrt(dot(z, z))
Base.dot(a::MVector{X, N}, b::MVector{X, N}) where {X, N} =
    sum(a.T.*b.T) + (a.s*b.s) + sum(dot.(a.x, b.x))

# define stuff necessary to use . notation with MVector
_get_seed(z::MVector, i::Int) = z.x[i]
_get_T(z::MVector) = z.T
_get_s(z::MVector) = z.s
_get_seed(z, i) = z
_get_T(z) = z
_get_s(z) = z

@generated function Base.Broadcast.broadcast!(f, dest::MVector{X, N}, 
                                           args::Vararg{Any, n}) where {X, N, n}
    expr = quote
        dest.T = broadcast(f, map(_get_T, args)...)
        dest.s = broadcast(f, map(_get_s, args)...)
    end
    for i = 1:N
        rhs = [:(_get_seed(args[$j], $i)) for j = 1:n]
        unshift!(expr.args, # push front
            :(broadcast!(f, _get_seed(dest, $i), $(rhs...))))
    end
    unshift!(expr.args, Expr(:meta, :inline))
    return expr
end