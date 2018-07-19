export GradientCache, 
       fromvector!,
       tovector!, 
       tovector

# ////// METHODS THAT CUSTOM TYPES NEED TO OVERLOAD //////
tovector!(out::Vector, ∇ᵤJ::Any, ∇ₜJ::Real, ∇ₛJ::Real) =
    throw(ArgumentError("not implemented"))

tovector(∇ᵤJ::Any, ∇ₜJ::Real, ∇ₛJ::Real) =
    throw(ArgumentError("not implemented"))

fromvector!(∇ᵤJ::Any, out::Vector) = 
    throw(ArgumentError("not implemented"))

# associated helper function
_checksize(out, ∇ᵤJ) =
    length(out) == length(∇ᵤJ) + 2 || 
            throw(ArgumentError("inconsistent input size"))


# ////// ACTUAL TYPE FOR CALCULATIONS //////
struct GradientCache{P, F, D, R, X} 
      ϕ::P # nonlinear propagator
  dϕdt!::F # nonlinear equation right hand side
   ddx!::D # x derivative
     ψ⁺::R # linearised adjoint propagator
    TMP::NTuple{5, X} # temporaries
end

# Constructor. Pass U as extra parameter for creating the temporaries
function GradientCache(ϕ, dϕdt!, ddx!, ψ⁺, U)

    # construct callable object
    cache = GradientCache(ϕ, dϕdt!, ddx!, ψ⁺, ntuple(i->similar(U), 5))

    # preallocate buffer, with space for cost function and gradient
    buffer = (zeros(1), tovector(U, 0, 0))

    # use Optim.jl trick for shared calculations
    last_x = similar(buffer[2])

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

function (agc::GradientCache)(x::AbstractVector, buffer)
    # ~~~ UNPACK LOCAL VARIABLES AND OBTAIN ALIASES ~~~
    U₀, T, s = fromvector!(agc.TMP[1], x)
    χ, ∇ᵤJ, Uₜ, dUₜdt = agc.TMP[2:end]
    ϕ        = agc.ϕ
    dϕdt!    = agc.dϕdt!
    ddx!     = agc.ddx!
    ψ⁺       = agc.ψ⁺
    
    # ~~~ GRADIENT WRT TO INITIAL CONDITION ~~~
    # obtain shifted end point
    Uₜ .= U₀
    shift!(ϕ(Uₜ, (0, T)), s)

    # obtain end point acceleration
    dϕdt!(T, Uₜ, dUₜdt)

    # obtain end point error
    χ .= Uₜ .- U₀

    # store cost to buffer
    buffer[1][1] = 0.5*dot(χ, χ)

    # calculate gradient WRT to period
    ∇ₜJ = dot(χ, dUₜdt)

    # calculate gradient WRT to shift
    ∇ₛJ = dot(χ, ddx!(Uₜ))

    # copy, reverse shift, backward integrate, subtract to obtain gradient.
    Uₜ .= χ
    ∇ᵤJ .= ψ⁺(shift!(Uₜ, -s), (T, 0)) .- χ

    # now obtain the other two gradients
    tovector!(buffer[2], ∇ᵤJ, ∇ₜJ, ∇ₛJ)

    return nothing
end