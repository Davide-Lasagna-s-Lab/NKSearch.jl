# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

using Flows: AbstractStageCache, reset!
# using Iterators

export GradientCache, 
       fromvector!,
       tovector!, 
       tovector

# ////// METHODS THAT CUSTOM TYPES NEED TO OVERLOAD //////
tovector!(out, dJdU0_L, dJdU0_R, ∇TLJ, ∇TRJ, ∇sJ) =
    throw(ArgumentError("not implemented"))

tovector(dJdU0_L, dJdU0_R, ∇TLJ, ∇TRJ, ∇sJ) =
    throw(ArgumentError("not implemented"))

fromvector!(dJdU0_L, dJdU0_R, out) = 
    throw(ArgumentError("not implemented"))

# associated helper function
_checksize(out, dJdU0_L, dJdU0_R) =
    length(out) == length(dJdU0_L) + length(dJdU0_R) + 3 || 
            throw(ArgumentError("inconsistent input size"))


# ////// ACTUAL TYPE FOR CALCULATIONS //////
struct GradientCache{P, F, D, R, X, C} 
      ϕ::P             # nonlinear propagator
  dϕdt!::F             # nonlinear equation right hand side
   ddx!::D             # x derivative
     ψ⁺::R             # linearised adjoint propagator
 caches::Tuple{C, C}   # two stage caches filled by the nonlinear operator
   U0_C::X             # right initial condition
   UF_C::X             # left terminal condition
    TMP::NTuple{10, X} # temporaries
end

# constructor
function GradientCache(ϕ, dϕdt!, ddx!, ψ⁺, 
                       U0_C::X, UF_C::X,
                       caches::Tuple{C, C}) where {X, C<:AbstractStageCache}

    # construct callable object
    cache = GradientCache(ϕ, 
                          dϕdt!, 
                          ddx!, 
                          ψ⁺, 
                          caches,
                          U0_C,
                          UF_C,
                          ntuple(i->deepcopy(U0_C), 10))

    # preallocate buffer, with space for cost function and gradient
    buffer = (zeros(1), tovector(U0_C, UF_C, 0, 0, 0))

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

function (agc::GradientCache)(x::AbstractVector, buffer)
    # ~~~ UNPACK LOCAL VARIABLES AND OBTAIN ALIASES ~~~
    U0_L, U0_R, TL, TR, s = fromvector!(agc.TMP[1], agc.TMP[2], x)
    χ_L, χ_R, χ_C, UF_L, UF_R, TMP, dJdU0_L, dJdU0_R = agc.TMP[3:end]
    U0_C   = agc.U0_C
    UF_C   = agc.UF_C
    ϕ      = agc.ϕ
    dϕdt!  = agc.dϕdt!
    ddx!   = agc.ddx!
    ψ⁺     = agc.ψ⁺
    caches = agc.caches

    # ~~~ FINAL POINTS AND ERRORS ~~~
    # χ_L
    UF_L .= U0_L
    ϕ(UF_L, (0, TL), reset!(caches[2]))
    χ_L .= UF_L .- U0_C

    # χ_R
    UF_R .= U0_R
    shift!(ϕ(UF_R, (0, TR), reset!(caches[1])), s)
    χ_R .= UF_R .- U0_L

    # χ_C
    χ_C .= UF_C .- U0_R

    # ~~~ TOTAL COST ~~~
    buffer[1][1] = (dot(χ_C, χ_C) + dot(χ_R, χ_R) + dot(χ_L, χ_L))/2


    # ~~~ GRADIENT WRT INITIAL CONDITIONS ~~~
    # gradient wrt U0_L
    TMP .= χ_L
    dJdU0_L .= ψ⁺(TMP, caches[2]) .- χ_R

    # and wrt U0_R
    TMP .= χ_R
    dJdU0_R .= ψ⁺(shift!(TMP, -s), caches[1]) .- χ_C


    # ~~~ GRADIENT WRT TIME AND SHIFT ~~~
    # TL
    ∇TLJ = dot(χ_L, dϕdt!(0, UF_L, TMP))

    # TR
    ∇TRJ = dot(χ_R, dϕdt!(0, UF_R, TMP))

    # s
    ∇sJ = dot(χ_R, ddx!(UF_R))


    # ~~~ PACK AND RETURN ~~~
    # now obtain the other two gradients
    tovector!(buffer[2], dJdU0_L, dJdU0_R, ∇TLJ, ∇TRJ, ∇sJ)

    return nothing
end
