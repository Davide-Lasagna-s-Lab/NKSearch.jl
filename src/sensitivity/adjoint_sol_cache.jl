# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import Base.Threads: threadid, @threads, nthreads
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

# Main outer constructor
function make_adjoint_problem(z::MVector{X, N, NS},
                              store,
                              L,
                              J,
                              S,
                              D,
                            jTJ::Real) where {X, N, NS}

    # make copies of the propagators
    Js = ntuple(i->deepcopy(J), nthreads())
    Ls = ntuple(i->deepcopy(L), nthreads())

    # temporaries
    tmp = similar(z[1])

    # period and shift
    T, s = z.d

    # various points on the orbit
     x0   =  store(similar(z[1]), 0, Val(0))
     xT   =  store(similar(z[1]), T, Val(0))
    dxTdT =  store(similar(z[1]), T, Val(1))

    # right hand side
    rhs = similar(z)

    @threads for i = 1:N
        # note that store must be thread safe
        id = threadid()

        # integration span
        span = (T - (i-1)*T/N, T - i*T/N)

        # set homogeneus initial condition
        rhs[i] .= 0

        # propagate
        Js[id](rhs[i], store, span)

        # flip sign
        rhs[i] .*= -1.0
    end

    # shift the last state
    S(rhs[N], -s)

    # set the last bits
    vals = zeros(NS); vals[1] = jTJ
    rhs.d = tuple(vals...)

    # construct object
    return AdjointProblemLHS(Ls, S, D, x0, xT, dxTdT, tmp, z, store), rhs
end

# Main interface is matrix-vector product exposed to the Krylov solver
Base.:*(A::AdjointProblemLHS{X}, w::MVector{X}) where {X} = mul!(similar(w), A, w)

# Compute mat-vec product (version including one spatial shifts)
function mul!(out::MVector{X, N, 2},
               mm::AdjointProblemLHS{X, N, 2},
                w::MVector{X, N, 2}) where {X, N}
    # aliases
    store  = mm.store
    x0     = mm.x0
    xT     = mm.xT
    dxTdT  = mm.dxTdT
    Ls     = mm.Ls
    D      = mm.D
    S      = mm.S
    z      = mm.z
    tmp    = mm.tmp
    T, s   = mm.z.d

    # main block
    @threads for i = 1:N
        id = threadid()

        # set adjoint final condition
        out[i] .= w[i]

        # integration span
        span = (T - (i-1)*T/N, T - i*T/N)

        # integrate linearised equations
        Ls[id](out[i], store, span)

        # apply shift on last segment
        i == N && S(out[i], -s)

        # this is the identity operator
        out[i] .-= w[i%N + 1]
    end

    # right columns
    out[N] .-= S(D[1](tmp, x0), -s).*w.d[1]
    out[N] .-= S(D[2](tmp, x0), -s).*w.d[2]

    # bottom rows
    out.d = (dot(w[1], dxTdT), dot(w[1], D[2](tmp, xT)))

    return out
end


# # Compute mat-vec product (version for no shifts)
# function mul!(out::MVector{X, N, 1},
#                 A::AdjointProblemLHS{X, N, 1},
#                 w::MVector{X, N, 1}) where {X, N}
#     # aliases
#     xT    = A.xT
#     Ls    = A.Ls
#     D     = A.D
#     z     = A.z
#     tmps   = A.tmps
#     dxTdTs = A.dxTdTs

#     # main block
#     @threads for i = 1:N
#         out[i] .= w[i]
#         Ls[threadid()](out[i], caches[i])
#         out[i] .-= w[(i+N-2)%N + 1]
#     end

#     # right column
#     out[1] .+= D[1](tmps, z[1]).*w.d[1]

#     # bottom row
#     out.d = (sum(dot(w[j], dxTdTs[j]) for j = 1:N)/N, )

#     return out
# end