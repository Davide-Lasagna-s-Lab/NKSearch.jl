# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import Base.Threads: @sync, @spawn
import LinearAlgebra: dot
import Flows

export make_adjoint_problem

# ~~~ Matrix Type ~~~
struct AdjointProblemLHS{X, N, NS, LST, ST, DT, CT}
        Ls::LST               # homogeneous adjoint operators (one per thread)
         S::ST                # space shift operator
         D::DT                # time (and space) derivative operator
        x0::X                 # initial point
        xT::X                 # final point
     dxTdT::X                 # time derivative of flow operator
       tmp::X                 # temporary storage
         z::MVector{X, N, NS} # the periodic orbit
     store::CT                # store
end

# TODO: test interface with no shift
# Main outer constructor
function make_adjoint_problem(z::MVector{X, N, NS},
                              store,
                              L,
                              J,
                              S,
                              D,
                            jTJ::Real) where {X, N, NS}

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
        @spawn begin
            # note that store must be thread safe
            # integration span
            span = (T - (i-1)*T/N, T - i*T/N)

            # set homogeneus initial condition
            rhs[i] .= 0

            # propagate
            Js[i](rhs[i], store, span)

            # flip sign
            rhs[i] .*= -1.0
        end
    end

    # shift the last state
    NS == 2 && S(rhs[N], -s)

    # set the last bits
    vals = zeros(NS); vals[1] = jTJ
    rhs.d = tuple(vals...)

    # construct object
    return AdjointProblemLHS(Ls, S, D, x0, xT, dxTdT, tmp, z, store), rhs
end

# outer constructor without shift
make_adjoint_problem(z::MVector{X, N, 1},
                     store,
                     L,
                     J,
                     D,
                     jTJ) where {X, N} = make_adjoint_problem(z,
                                                              store,
                                                              L,
                                                              J,
                                                              nothing,
                                                              D,
                                                              jTJ)

# Main interface is matrix-vector product exposed to the Krylov solver
Base.:*(A::AdjointProblemLHS{X}, w::MVector{X}) where {X} = mul!(similar(w), A, w)

# Compute mat-vec product (version including one spatial shifts)
function mul!(out::MVector{X, N, NS}, mm::AdjointProblemLHS{X, N, NS}, w::MVector{X, N, NS}) where {X, N, NS}
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
        @spawn begin
            # set adjoint final condition
            out[i] .= w[i]

            # integration span
            span = (T - (i-1)*T/N, T - i*T/N)

            # integrate linearised equations
            Ls[i](out[i], store, span)

            # apply shift on last segment
            NS == 2 && i == N && S(out[i], -s)

            # this is the identity operator
            out[i] .-= w[i%N + 1]
        end
    end

    # right columns
    out[N] .-= NS == 2 ? S(D[1](tmp, x0), -s).*w.d[1] : D[1](tmp, x0).*w.d[1]
    NS == 2 && (out[N] .-= S(D[2](tmp, x0), -s).*w.d[2])

    # bottom rows
    out.d = ntuple(j-> j == 1 ? dot(w[1], dxTdT) : dot(w[1], D[j](tmp, xT)), length(D))

    return out
end
