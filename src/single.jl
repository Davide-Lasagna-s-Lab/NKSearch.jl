export ADJGradCache

# ////// METHODS THAT CUSTOM TYPES NEED TO OVERLOAD //////
tovector!(out::Vector, ∇ᵤJ::Any, ∇ₜJ::Real, ∇ₛJ::Real) =
    throw(ArgumentError("not implemented"))
fromvector(∇ᵤJ::Any, out::Vector) = 
    throw(ArgumentError("not implemented"))

# associated helper function
_checksize(out, ∇ᵤJ) =
    length(out) == length(∇ᵤJ) + 2 || 
            throw(ArgumentError("inconsistent input size"))


# ////// ACTUAL TYPE FOR CALCULATIONS //////
struct ADJGradCache{P, F, M, R, X} 
       ϕ::P # nonlinear propagator
       F::F # nonlinear equation right hand side
     mon::M # monitor for forward solution
      ψ⁺::R # linearised adjoint propagator
    TMP1::X # temporary
    TMP2::X # temporary
    TMP3::X # temporary
    TMP4::X # temporary
end

# constructor. Pass U as extra parameter for creating the temporaries
ADJGradCache(ϕ, F, mon, ψ⁺, U) = 
    ADJGradCache(ϕ, F, mon, ψ⁺, [similar(U) for i=1:4]...) 

function (agc::ADJGradCache)(grad::AbstractVector, x::AbstractVector)
    # ~~~ UNPACK LOCAL VARIABLES AND OBTAIN ALIASES ~~~
    U₀, T, s = fromvector!(agc.TMP1, x)
    χ        = agc.TMP2
    ∇ᵤJ      = agc.TMP3
    dU₁dt    = agc.TMP4
    ϕ        = agc.ϕ
    F        = agc.F
    ψ⁺       = agc.ψ⁺
    mon      = agc.mon
    
    # ~~~ GRADIENT WRT TO INITIAL CONDITION ~~~
    # obtain end point
    Uₜ .= U₀
    ϕ(Uₜ, (0, T), reset!(mon))

    # obtain end point error
    χ .= Uₜ .- U₀

    # obtain end point acceleration
    F(T, Uₜ, dUₜdt)

    # calculate gradient WRT to period
    ∇ₜJ = dot(χ, shift!(dU₁dt, s))

    # calculate gradient WRT to shift
    ∇ₛJ = dot(χ, ddx!(U₁))

    # reverse shift, backward integrate, subtract to obtain gradient.
    ∇ᵤJ .= ψ⁺(shift!(χ, -s), (T, 0)) .- χ

    # now obtain the other two gradients
    return tovector!(grad, ∇ᵤJ, ∇ₜJ, ∇ₛJ)
end