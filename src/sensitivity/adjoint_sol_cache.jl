# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import Base.Threads: @sync, @spawn
import LinearAlgebra: dot

export make_adjoint_problem

# ~~~ Matrix Type ~~~
struct AdjointProblemLHS{X, N, NS, ORDERING, LST, ST, DT, CT}
        Ls::LST               # homogeneous adjoint operators (one per thread)
         S::ST                # space shift operator
         D::DT                # time (and space) derivative operator
        x0::X                 # initial point
        xT::X                 # final point
     dxTdT::X                 # time derivative of flow operator
       tmp::X                 # temporary storage
         z::MVector{X, N, NS} # the periodic orbit
     store::CT                # store

    AdjointProblemLHS{X, N, NS, ORDERING}(Ls::LST,
                                           S::ST,
                                           D::DT,
                                          x0::X,
                                          xT::X,
                                       dxTdT::X,
                                         tmp::X,
                                           z::MVector{X, N, NS},
                                       store::CT) where {X, N, NS, ORDERING, LST, ST, DT, CT} =
        new{X, N, NS, ORDERING, LST, ST, DT, CT}(Ls, S, D, x0, xT, dxTdT, tmp, z, store)
end

"""
    make_adjoint_problem(z, store, L, J, S, D, jTJ) -> (A, rhs)
    make_adjoint_problem(z, store, L, J,    D, jTJ) -> (A, rhs)   # no shift

Assemble the linear system for the **adjoint** sensitivity of a converged
(relative) periodic orbit `z`. Returns a matrix-free operator `A` (callable
as `A * w` and usable with GMRES) together with its right-hand side `rhs`.

# Arguments
- `z::MVector`: a *converged* orbit (see [`search!`](@ref)).
- `store`: a thread-safe trajectory cache that returns the orbit state and
  its time derivative at a requested time, callable as `store(out, t, Val(k))`
  for derivative order `k`.
- `L`: homogeneous adjoint flow operator.
- `J`: inhomogeneous (forcing) adjoint operator.
- `S`, `D`: spatial-shift operator and the time/space derivative operators,
  as for [`search!`](@ref) (`S` omitted in the no-shift form).
- `jTJ::Real`: the cost-gradient scalar placed in the bottom rows of `rhs`.
- `row_order::Symbol`: arrangement of rows used for the linear system, either
  `:ashtari` or `:regular`.

!!! warning "Experimental"
    The sensitivity interface is still being stabilised; argument
    conventions (especially the `store` and adjoint-operator contracts) may
    change, and the no-shift path is not yet covered by tests. The companion
    tangent problem is built by the (currently unexported)
    `NKSearch.make_tangent_problem`.
"""
function make_adjoint_problem(z::MVector{X, N, NS},
                              store,
                              L,
                              J,
                              S,
                              D,
                            jTJ::Real;
                      row_order::Symbol=:ashtari) where {X, N, NS}
    row_order ∈ (:ashtari, :regular) || throw(ArgumentError("invalid argument: `row_order`, only values `:ashtari` or `:regular` are permitted"))

    # make copies of the propagators, one per segment (see the threading
    # note in newton.jl on why we avoid threadid()-indexed buffers)
    Js = ntuple(i->deepcopy(J), N)
    Ls = ntuple(i->deepcopy(L), N)

    # temporaries
    tmp = similar(z[1])

    # period and shift
    if NS == 2
        T, s = z.d
    else
        T,   = z.d
    end

    # various points on the orbit
     x0   =  store(similar(z[1]), 0, Val(0))
     xT   =  store(similar(z[1]), T, Val(0))
    dxTdT =  store(similar(z[1]), T, Val(1))

    # right hand side
    rhs = similar(z)

    @sync for i = 1:N
        j = row_order == :ashtari ? i%N + 1 : i
        @spawn begin
            # note that store must be thread safe
            # integration span
            span = (T - (i-1)*T/N, T - i*T/N)

            # set homogeneus initial condition
            rhs[j] .= 0

            # propagate
            Js[i](rhs[j], store, span)

            # flip sign
            rhs[j] .*= -1.0
        end
    end

    # shift the last state
    NS == 2 && S(rhs[row_order == :ashtari ? 1 : N], -s)

    # set the last bits
    vals = zeros(NS); vals[1] = jTJ
    rhs.d = tuple(vals...)

    # construct object
    return AdjointProblemLHS{X, N, NS, row_order}(Ls, S, D, x0, xT, dxTdT, tmp, z, store), rhs
end

# outer constructor without shift
make_adjoint_problem(z::MVector{X, N, 1},
                     store,
                     L,
                     J,
                     D,
                     jTJ;
             row_order::Symbol=:ashtari) where {X, N} = make_adjoint_problem(z,
                                                                             store,
                                                                             L,
                                                                             J,
                                                                             nothing,
                                                                             D,
                                                                             jTJ;
                                                                       row_order=row_order)

# Main interface is matrix-vector product exposed to the Krylov solver
Base.:*(A::AdjointProblemLHS{X}, w::MVector{X}) where {X} = mul!(similar(w), A, w)

# Compute mat-vec product (version including one spatial shifts)
function mul!(out::MVector{X, N, NS},
               mm::AdjointProblemLHS{X, N, NS, ORDERING},
                w::MVector{X, N, NS}) where {X, N, NS, ORDERING}
    # aliases
    store = mm.store
    x0    = mm.x0
    xT    = mm.xT
    dxTdT = mm.dxTdT
    Ls    = mm.Ls
    D     = mm.D
    S     = mm.S
    z     = mm.z
    tmp   = mm.tmp
    T     = mm.z.d[1]
    s     = NS == 2 ? mm.z.d[2] : 0.0

    # main block
    @sync for i = 1:N
        j = ORDERING == :ashtari ? i%N + 1 : i
        @spawn begin
            # set adjoint final condition
            out[j] .= w[i]

            # integration span
            span = (T - (i-1)*T/N, T - i*T/N)

            # integrate linearised equations
            Ls[i](out[j], store, span)

            # apply shift on last segment
            NS == 2 && i == N && S(out[j], -s)

            # this is the identity operator
            out[j] .-= w[i%N + 1]
        end
    end

    # right columns
    out[ORDERING == :ashtari ? 1 : N] .-= NS == 2 ? S(D[1](tmp, x0), -s).*w.d[1] : D[1](tmp, x0).*w.d[1]
    NS == 2 && (out[ORDERING == :ashtari ? 1 : N] .-= S(D[2](tmp, x0), -s).*w.d[2])

    # bottom rows
    out.d = ntuple(j-> j == 1 ? dot(w[1], dxTdT) : dot(w[1], D[j](tmp, xT)), length(D))

    return out
end
