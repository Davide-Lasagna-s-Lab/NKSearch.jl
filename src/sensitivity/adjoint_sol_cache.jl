# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export make_adjoint_problem

# ~~~ Matrix Type ~~~
struct AdjointProblemLHS{X, N, NS, ORDERING, LST, ST, DT}
        Ls::LST               # homogeneous adjoint operators (one per thread)
         S::ST                # space shift operator
         D::DT                # time (and space) derivative operator
        x0::X                 # initial point
        xT::X                 # final point, u(T) = S(u(0), -s) on a converged orbit
       tmp::X                 # temporary storage
         z::MVector{X, N, NS} # the periodic orbit

    AdjointProblemLHS{X, N, NS, ORDERING}(Ls::LST,
                                           S::ST,
                                           D::DT,
                                          x0::X,
                                          xT::X,
                                           z::MVector{X, N, NS}) where {X, N, NS, ORDERING, LST, ST, DT} =
        new{X, N, NS, ORDERING, LST, ST, DT}(Ls, S, D, x0, xT, similar(z[1]), z)
end

function _make_adjoint_rhs(Js, S, s, z::MVector{X, N, NS}, jTJ, row_order) where {X, N, NS}
    # right hand side
    rhs = similar(z)

    @sync for i = 1:N
        j = row_order == :ashtari ? i%N + 1 : i
        @spawn begin
            # set homogeneus initial condition
            rhs[j] .= 0

            # propagate
            Js[i](rhs[j])

            # flip sign
            rhs[j] .*= -1.0
        end
    end

    # shift the last state
    NS > 1 && S(rhs[row_order == :ashtari ? 1 : N], .-s)

    # set the last bits
    vals = zeros(NS); vals[1] = jTJ
    rhs.d = tuple(vals...)

    return rhs
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
  for derivative order `k`, or a tuple of one `RAMStageCache` per segment. For
  a relative periodic orbit a `RAMStorage` must not be built with a `period`:
  its wrap-around assumes u(t + T) = u(t), whereas u(0) = S(u(T), s). Fill it
  segment by segment with `storelast=false` and push the end point u(T).
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
                          store::Flows.RAMStorage,
                              L,
                              J,
                              S,
                              D,
                            jTJ::Real;
                      row_order::Symbol=:ashtari) where {X, N, NS}
    row_order ∈ (:ashtari, :regular) || throw(ArgumentError("invalid argument: `row_order`, only values `:ashtari` or `:regular` are permitted"))

    # period and shift
    T = z.d[1]
    s = NS > 1 ? Base.tail(z.d) : ()

    # make copies of the propagators, one per segment (see the threading
    # note in newton.jl on why we avoid threadid()-indexed buffers)
    Ls = ntuple(i -> let Li = deepcopy(L)
        (out) -> Li(out, store, (T - (i - 1)*T/N - 1e-12, T - i*T/N)) # subtract small amount to avoid timespan error resulting from floating point arithmetic
    end, N)
    Js = ntuple(i -> let Ji = deepcopy(J)
        (out) -> Ji(out, store, (T - (i - 1)*T/N - 1e-12, T - i*T/N))
    end, N)

    # the first point of the orbit, and its last point, u(T) = S(u(0), -s) on a converged orbit (no value of the
    # store at T is needed)
    x0 = store(similar(z[1]), 0, Val(0))
    xT = NS > 1 ? S(copy(x0), .-s) : copy(x0)

    # right-hand side
    rhs = _make_adjoint_rhs(Js, S, s, z, jTJ, row_order)

    return AdjointProblemLHS{X, N, NS, row_order}(Ls, S, D, x0, xT, z), rhs
end

function make_adjoint_problem(z::MVector{X, N, NS},
                          store::NTuple{N, Flows.RAMStageCache},
                              L,
                              J,
                              S,
                              D,
                            jTJ::Real;
                      row_order::Symbol=:ashtari) where {X, N, NS}
    row_order ∈ (:ashtari, :regular) || throw(ArgumentError("invalid argument: `row_order`, only values `:ashtari` or `:regular` are permitted"))

    # shift
    s = NS > 1 ? Base.tail(z.d) : ()

    # make copies of the propagators, one per segment (see the threading
    # note in newton.jl on why we avoid threadid()-indexed buffers)
    Ls = ntuple(i -> let Li = deepcopy(L)
        (out) -> Li(out, store[N - i + 1])
    end, N)
    Js = ntuple(i -> let Ji = deepcopy(J)
        (out) -> Ji(out, store[N - i + 1])
    end, N)

    # the first point of the orbit, and its last point, u(T) = S(u(0), -s) on a converged orbit
    x0 = copy(store[1].xs[1][1])
    xT = NS > 1 ? S(copy(x0), .-s) : copy(x0)

    # right-hand side
    rhs = _make_adjoint_rhs(Js, S, s, z, jTJ, row_order)

    return AdjointProblemLHS{X, N, NS, row_order}(Ls, S, D, x0, xT, z), rhs
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
Base.:*(A::AdjointProblemLHS{X}, w::MVector{X}) where {X} = LinearAlgebra.mul!(similar(w), A, w)

# Compute mat-vec product (version including one spatial shifts)
function LinearAlgebra.mul!(out::MVector{X, N, NS},
                             mm::AdjointProblemLHS{X, N, NS, ORDERING},
                              w::MVector{X, N, NS}) where {X, N, NS, ORDERING}
    # aliases
    x0    = mm.x0
    xT    = mm.xT
    Ls    = mm.Ls
    D     = mm.D
    S     = mm.S
    tmp   = mm.tmp
    NS > 1 && (s = Base.tail(mm.z.d))

    # main block
    @sync for i = 1:N
        j = ORDERING == :ashtari ? i%N + 1 : i
        @spawn begin
            # set adjoint final condition
            out[j] .= w[i]

            # integrate linearised equations
            Ls[i](out[j])

            # apply shift on last segment
            NS > 1 && i == N && S(out[j], .-s)

            # this is the identity operator
            out[j] .-= w[i%N + 1]
        end
    end

    # right columns
    if NS > 1
        for i in eachindex(D)
            out[ORDERING == :ashtari ? 1 : N] .-= S(D[i](tmp, x0), .-s).*w.d[i]
        end
    else
        out[ORDERING == :ashtari ? 1 : N] .-= D[1](tmp, x0).*w.d[1]
    end

    # bottom rows
    out.d = ntuple(i -> dot(w[1], D[i](tmp, xT)), length(D))

    return out
end
