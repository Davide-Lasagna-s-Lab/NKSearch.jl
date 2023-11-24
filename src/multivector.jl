# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

import LinearAlgebra
import HDF5: write, h5open, h5readattr, attrs

export MVector, 
       tovector,
       fromvector!,
       nsegments,
       save_seeds,
       load_seeds!,
       find_number_of_segments

# ~~~ INTERFACE FOR MVector and MMatrix ~~~
# The type parameter `X` must support
# 1) dot(::X, ::X)
# 2) similar(::X)
# 3) full broadcast functionality, with variables of type `X` and scalars

# ~~~ Vector Type ~~~
mutable struct MVector{X, N, NS} <: AbstractVector{Float64}
    x::NTuple{N, X}        # the seed along the orbit
    d::NTuple{NS, Float64} # period and optional NS-1 shifts
    MVector(x::NTuple{N, X}, T::Real, s::Real) where {N, X} =
        new{X, N, 2}(x, Float64.((T, s)))
    MVector(x::NTuple{N, X}, T::Real) where {N, X} =
        new{X, N, 1}(x, Float64.((T, )))
end

# getindex to have z[i] mean z.x[i]
Base.getindex(z::MVector, i::Int) = z.x[i]

# number of segments
nsegments(::MVector{X, N}) where {X, N} = N

# interface for GMRES solver
Base.similar(z::MVector) = MVector(similar.(z.x), z.d...)
Base.copy(z::MVector) = MVector(copy.(z.x), z.d...)
LinearAlgebra.norm(z::MVector) = sqrt(LinearAlgebra.dot(z, z))
LinearAlgebra.dot(a::MVector{X, N}, b::MVector{X, N}) where {X, N} =
    sum(a.d.*b.d) + sum(LinearAlgebra.dot.(a.x, b.x))

# define stuff necessary to use . notation with MVector (only works
# for short! expressions) otherwise code is very slow!
const MVectorStyle = Broadcast.ArrayStyle{MVector}
Base.BroadcastStyle(::Type{<:MVector}) = Broadcast.ArrayStyle{MVector}()
Base.BroadcastStyle(::Broadcast.ArrayStyle{MVector},
                    ::Broadcast.DefaultArrayStyle{1}) = Broadcast.DefaultArrayStyle{1}()
Base.BroadcastStyle(::Broadcast.DefaultArrayStyle{1},
                    ::Broadcast.ArrayStyle{MVector}) = Broadcast.DefaultArrayStyle{1}()

# "Linearise" the data structure into a single Float64 vector
function tovector(z::MVector{X, N, NS}) where {X, N, NS}
    n = length(z[1])
    out = zeros(N*n + NS)
    for i = 1:N
        out[_blockrng(i, n)] .= z[i]
    end
    out[end - NS + 1 : end] .= z.d
    return out
end

# Copy the data in the vector v into a tuple of object structure of MVector
function fromvector!(out::MVector{X, N, NS}, v::Vector{<:Real}) where {X, N, NS}
    n = length(out[1])
    for i = 1:N
        out[i] .= view(v, _blockrng(i, n))
    end
    out.d = ntuple(j->v[end-NS+j], NS)
    return out
end

# a hack really!
Base.size(z::MVector{X, N, NS}) where {X, N, NS} = (NS + N*length(z.x[1]), )

# getters
_get_seed(z::MVector, i) = z.x[i]
_get_seed(z, i) = z
_get_d(z::MVector) = z.d
_get_d(z) = z

@inline function Broadcast.copyto!(dest::MVector{X, N},
                                     bc::Broadcast.Broadcasted{MVectorStyle}) where {X, N}
    bcf = Broadcast.flatten(bc)
    for i = 1:N
        Broadcast.broadcast!(bcf.f, dest.x[i], map(arg->_get_seed(arg, i), bcf.args)...)
    end
    dest.d = Broadcast.broadcast(bcf.f, map(_get_d, bcf.args)...)
    return dest
end

function save(z::MVector{X, N, NS}, path::String) where {X, N, NS}
    # save trajectory to a large matrix first
    data = zeros(Float64, length(z[1]), N)
    for (i, zi) in enumerate(z.x)
        data[:, i] .= zi
    end
    h5open(path, "w") do file
        write(file, "seed", data)
        for i = 1:NS
            attrs(file)["d$i"] = z.d[i]
        end
    end
end

# Return integer N such that M/N is integer, where T/Tmin < N < T/Tmax.
# Return 0 if no such integer can be found
function find_number_of_segments(M::Int, T::Real, Tmin::Real, Tmax::Real)
    Nmin = Int(floor(T/Tmin))
    Nmax = Int(ceil( T/Tmax))    
    for N = Nmax:Nmin
        if M % N == 0 
            return N
        end
    end
    return 0
end

# a hack
_is_complex_eltype(z::MVector) = eltype(parent(z[1])) <: Complex

# save MVector to file, including other 
function save_seeds(z::MVector{X, N, NS},
                 path::String,
                other::Dict{String, <:Any} = Dict{String, Any}()) where {X, N, NS}
    # test whether X is a complex type
    h5open(path, "w") do file
        if _is_complex_eltype(z)
            for i = 1:N
                write(file, "seed_$(i)_real", real(parent(z.x[i])))
                write(file, "seed_$(i)_imag", imag(parent(z.x[i])))
            end
        else
            for i = 1:N
                write(file, "seed_$i", parent(z.x[i]))
            end
        end
        for i = 1:NS
            attrs(file)["d$i"] = z.d[i]
        end
        for (k, v) in other
            attrs(file)["other_$k"] = v
        end
    end
end

# load data and return a NamedTuple of other attributes that were stored
function load_seeds!(fun, path::String)
    h5open(path, "r") do file
        
        # load attributes handle
        attributes = attrs(file)

        # load bit here
        dict = Dict{String, Any}()

        # store seeds here
        xs = []

        # determine if we saved complex data
        is_complex_data = "seed_1_real" in keys(file)
            
        # number of seeds
        N = is_complex_data ? div(length(keys(file)), 2) : length(keys(file))

        # read real and imaginary part if we need so
        if is_complex_data
            for i = 1:N
                push!(xs, fun(read(file, "seed_$(i)_real") .+ im.*read(file, "seed_$(i)_imag")))
            end
        else
            for i = 1:N
                push!(xs, read(file, "seed_$i"))
            end
        end
        
        # and period and shifts (all those that start with d)
        d = [attributes[el] for el in keys(attributes) if startswith(el, "d")]

        # also load other bits that might have been saved
        for k in keys(attributes)
            if startswith(k, "other_")
                dict[k[7:end]] = attributes[k]
            end
        end
       
        return MVector(tuple(xs...), d...), dict
    end
end