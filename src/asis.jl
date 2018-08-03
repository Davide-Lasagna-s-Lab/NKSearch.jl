module asis

using Flows: AbstractStageCache, reset!
using Iterators

export GradientCache, 
       fromvector!,
       tovector!, 
       tovector

# ////// METHODS THAT CUSTOM TYPES NEED TO OVERLOAD //////
tovector!(out::Vector, ∇ᵤJ::NTuple{N, X}, ∇ₜJ::Real, ∇ₛJ::Real) where {N, X}=
    throw(ArgumentError("not implemented"))

tovector(∇ᵤJ::NTuple{N, X}, ∇ₜJ::Real, ∇ₛJ::Real) where {N, X} =
    throw(ArgumentError("not implemented"))

fromvector!(∇ᵤJ::NTuple{N, X}, out::Vector) where {N, X} = 
    throw(ArgumentError("not implemented"))

# associated helper function
_checksize(out::Vector, ∇ᵤJ::NTuple{N, X}) where {N, X} =
    length(out) == sum(length.(∇ᵤJ)) + 2 || 
            throw(ArgumentError("inconsistent input size"))


# ////// ACTUAL TYPE FOR CALCULATIONS //////
mutable struct GradientCache{N, P, F, D, R, X, CS} 
      ϕ::P  # nonlinear propagator
  dϕdt!::F  # nonlinear equation right hand side
   ddx!::D  # x derivative
     ψ⁺::R  # linearised adjoint propagator
 caches::CS # a vector of stage caches filled by the nonlinear operator
    TMP::Tuple{NTuple{N, X}, NTuple{N, X}, NTuple{N, X}, X, X} # temporaries
      K::Int
end

# Constructor. Pass U as extra parameter for creating the temporaries
function GradientCache(ϕ, dϕdt!, ddx!, ψ⁺, 
                       U::NTuple{N, X}, 
                       caches::NTuple{N, C}, 
                       K::Int=N) where {N, X, C<:AbstractStageCache}


    # construct callable object
    cache = GradientCache(ϕ, 
                          dϕdt!, 
                          ddx!, 
                          ψ⁺, 
                          caches,
                          (deepcopy.(U), 
                           deepcopy.(U), 
                           deepcopy.(U),
                           deepcopy(U[1]), 
                           deepcopy(U[1])), K)

    # preallocate buffer, with space for cost function and gradient
    buffer = (zeros(1), tovector(U, 0, 0))

    # use Optim.jl trick for shared calculations
    last_x = similar(buffer[2]); last_x[1] = rand()

    # callables
    function f(x)
        x != last_x && (cache(x, buffer); last_x .= x)
        return buffer[1][1]
    end

    function g!(grad, x)
        x != last_x && (cache(x, buffer); last_x .= x)
        grad .= buffer[2]
        return nothing
    end

    # return the two closures
    return f, g!
end

# return k-1 with wrap around
_prev_k(k, N) = (k+N-2)%N + 1
_next_k(k, N) = (k+N)  %N + 1

function (agc::GradientCache{N})(x::AbstractVector, buffer) where {N}
    # ~~~ UNPACK LOCAL VARIABLES AND OBTAIN ALIASES ~~~
    U₀, T, s = fromvector!(agc.TMP[1], x)
    χ, ∇ᵤJ, Uₜ, dUₜdt = agc.TMP[2], agc.TMP[3], agc.TMP[4], agc.TMP[5]
    ϕ        = agc.ϕ
    dϕdt!    = agc.dϕdt!
    ddx!     = agc.ddx!
    ψ⁺       = agc.ψ⁺
    caches   = agc.caches

    # shifts for every segment
    sₖ = ntuple(i->(i == N ? s : zero(s)), N)

    # init data 
    buffer[1][1] = 0.0
    ∇ₜJ          = 0.0
    ∇ₛJ          = 0.0

    # set to zero gradients that will not be calculated
    for el in ∇ᵤJ; el .= 0; end
    for el in χ;   el .= 0; end
    
    # ~~~ GRADIENT WRT TO INITIAL CONDITION ~~~

    # this loop is not thread safe, because ϕ writes to the 
    # stage cache for ψ⁺ to read when integrating backwards. 
    # One should create copies of ϕ, making sure copies of 
    # the stage cache are done too. 
    ks = unique(collect(chain(1:agc.K, (N-agc.K+1):N)))

    for k in ks
        # obtain (optionally) shifted end point
        Uₜ .= U₀[k]
        shift!(ϕ(Uₜ, (0, T/N), reset!(caches[k])), sₖ[k])

        # obtain end point acceleration
        dϕdt!(T/N, Uₜ, dUₜdt)

        # calc error
        χ[k] .= Uₜ .- U₀[_next_k(k, N)]

        # now update gradient wrt to T
        ∇ₜJ += dot(χ[k], dUₜdt)/N

        # and update gradient WRT to shift
        if k == N
            ∇ₛJ += dot(χ[k], ddx!(Uₜ))
        end
    end
       
    for k in ks
        # update cost
        buffer[1][1] += 0.5*dot(χ[k], χ[k])

        # calculate gradient wrt to initial conditions, aliasing over Uₜ
        Uₜ .= χ[k]
        ∇ᵤJ[k] .= ψ⁺(shift!(Uₜ, -sₖ[k]), caches[k]) .- χ[_prev_k(k, N)]
    end

    # now obtain the other two gradients
    tovector!(buffer[2], ∇ᵤJ, ∇ₜJ, ∇ₛJ)

    return nothing
end

end