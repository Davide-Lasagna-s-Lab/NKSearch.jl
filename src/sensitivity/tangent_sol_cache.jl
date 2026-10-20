# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

# ! commit message:
# !  1. sensitivity code now accepts both RAMStorage and RAMStageCache
# !  2. subtracted offset when using RAMStorage to avoid timespan error from floating point arithmetic
# !  3. cleaned up imports and exports
# !  4. generalised code to accept arbitrary number of continuous shifts

# ~~~ Matrix Type ~~~
struct TangentProblemLHS{X, N, NS, ORDERING, LST, ST, DT}
       Ls::LST               # linearised flow operator(s)
        S::ST                # space shift operator
        D::DT                # time (and space) derivative operators
       x0::X                 # initial conditions
       xT::X                 # final conditions shifted back, S(u(T), s), which is u(0) on a converged orbit
      tmp::X                 # temporary storage
        z::MVector{X, N, NS} # current orbit

    TangentProblemLHS{X, N, NS, ORDERING}(Ls::LST,
                                           S::ST,
                                           D::DT,
                                          x0::X,
                                          xT::X,
                                           z::MVector{X, N, NS}) where {X, N, NS, ORDERING, LST, ST, DT} =
        new{X, N, NS, ORDERING, LST, ST, DT}(Ls, S, D, x0, xT, similar(z[1]), z)
end

function _make_tangent_rhs!(rhs::MVector{X, N, NS}, Js, S, s, row_order) where {X, N, NS}
    # compute inhomogeneous evolution for rhs
    @sync for i = 1:N
        j = row_order == :ashtari ? i%N + 1 : i
        @spawn begin
            # set homogeneous initial condition
            rhs[j] .= 0

            # integrate non-homogeneous equations over the i-th span
            Js[i](rhs[j])

            # flip sign for the right hand side
            rhs[j] .*= -1.0
        end
    end

    # we need to "back-shift" the last state it if we have a symmetry
    NS > 1 && S(rhs[row_order == :ashtari ? 1 : N], s)

    rhs.d = tuple(zeros(NS)...)

    return rhs
end

# Main outer constructor. For a relative periodic orbit the store must not be built with a `period`: its
# wrap-around assumes u(t + T) = u(t), whereas u(0) = S(u(T), s), so that the base flow near t = 0 and t = T would
# mix the two frames. Fill it segment by segment with `storelast=false` and push the end point u(T), or use the
# stage caches.
function make_tangent_problem(z::MVector{X, N, NS},
                          store::Flows.RAMStorage,
                              L,
                              J,
                              S,
                              D;
                      row_order::Symbol=:ashtari) where {X, N, NS}
    row_order ∈ (:ashtari, :regular) || throw(ArgumentError("invalid argument: `row_order`, only values `:ashtari` or `:regular` are permitted"))

    # period and shift
    T = z.d[1]
    s = NS > 1 ? Base.tail(z.d) : ()

    # make copies of the propagators, one per segment (see the threading
    # note in newton.jl on why we avoid threadid()-indexed buffers)
    Ls = ntuple(i -> let Li = deepcopy(L)
        (out) -> Li(out, store, ((i - 1)*T/N, i*T/N - 1e-12)) # subtract small amount to avoid timespan error resulting from floating point arithmetic
    end, N)
    Js = ntuple(i -> let Ji = deepcopy(J)
        (out) -> Ji(out, store, ((i - 1)*T/N, i*T/N - 1e-12))
    end, N)

    # the first point of the orbit, and its last point shifted back, S(u(T), s), which on a converged orbit is the
    # first point itself (no value of the store at T is needed)
    x0 = store(similar(z[1]), 0, Val(0))
    xT = copy(x0)

    # right-hand side
    rhs = _make_tangent_rhs!(similar(z), Js, S, s, row_order)

    return TangentProblemLHS{X, N, NS, row_order}(Ls, S, D, x0, xT, z), rhs
end

function make_tangent_problem(z::MVector{X, N, NS},
                          cache::NTuple{N, Flows.RAMStageCache},
                              L,
                              J,
                              S,
                              D;
                      row_order::Symbol=:ashtari) where {X, N, NS}
    row_order ∈ (:ashtari, :regular) || throw(ArgumentError("invalid argument: `row_order`, only values `:ashtari` or `:regular` are permitted"))

    # shift
    s = NS > 1 ? Base.tail(z.d) : ()

    # make copies of the propagators, one per segment (see the threading
    # note in newton.jl on why we avoid threadid()-indexed buffers)
    Ls = ntuple(i -> let Li = deepcopy(L)
        (out) -> Li(out, cache[i])
    end, N)
    Js = ntuple(i -> let Ji = deepcopy(J)
        (out) -> Ji(out, cache[i])
    end, N)

    # the first point of the orbit, and its last point shifted back, S(u(T), s), which on a converged orbit is the
    # first point itself
    x0 = copy(cache[1].xs[1][1])
    xT = copy(x0)

    # right-hand side
    rhs = _make_tangent_rhs!(similar(z), Js, S, s, row_order)

    return TangentProblemLHS{X, N, NS, row_order}(Ls, S, D, x0, xT, z), rhs
end

# outer constructor without shift
make_tangent_problem(z::MVector{X, N, 1},
                     store,
                     L,
                     J,
                     D;
             row_order::Symbol=:ashtari) where {X, N} = make_tangent_problem(z,
                                                                             store,
                                                                             L,
                                                                             J,
                                                                             nothing,
                                                                             D;
                                                                     row_order=row_order)

# Main interface is matrix-vector product exposed to the Krylov solver
Base.:*(mm::TangentProblemLHS{X}, w::MVector{X}) where {X} = mul!(similar(w), mm, w)

# Compute mat-vec product
function LinearAlgebra.mul!(out::MVector{X, N, NS},
                             mm::TangentProblemLHS{X, N, NS, ORDERING},
                              w::MVector{X, N, NS}) where {X, N, NS, ORDERING}
    # aliases
    x0    = mm.x0
    xT    = mm.xT
    Ls    = mm.Ls
    D     = mm.D
    S     = mm.S
    tmp   = mm.tmp
    NS > 1 && (s = Base.tail(mm.z.d))

    # compute L{x0[i]}⋅w[i] - w[i+1]
    @sync for i = 1:N
        j = ORDERING == :ashtari ? i%N + 1 : i
        @spawn begin
            # set perturbation initial condition
            out[j] .= w[i]

            # integrate linearised equations
            Ls[i](out[j])

            # apply shift on last segment
            NS > 1 && i == N && S(out[j], s)

            # this is the identity operators on the upper diagonal
            out[j] .-= w[i%N + 1]
        end
    end

    # Add the time and space derivatives at the end point shifted back,
    # S(u(T), s) = u(0): the columns of the period and of the shifts
    for i in eachindex(D)
        out[ORDERING == :ashtari ? 1 : N] .+= D[i](tmp, xT).*w.d[i]
    end

    # Compute phase locking constraints
    out.d = ntuple(i -> dot(w[1], D[i](tmp, x0)), length(D))

    return out
end
