# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import Base.Threads: threadid, @threads, nthreads
import LinearAlgebra: dot
import GMRES: gmres!
import Flows

# ~~~ Matrix Type ~~~
struct TangentProblemLHS{X, N, NS, LST, ST, DT, CT}
       Ls::LST               # linearised flow operator(s)
        S::ST                # space shift operator
        D::DT                # time (and space) derivative operators
       x0::X                 # initial conditions
       xT::X                 # final conditions
    dxTdT::X                 # time derivative of final point
      tmp::X                 # temporary storage
        z::MVector{X, N, NS} # current orbit
    store::CT                # store
end

# Main outer constructor
function make_tangent_problem(z::MVector{X, N, NS},
                              store,
                              L,
                              J,
                              S,
                              D) where {X, N, NS}

    # make copies of the propagators, one for each thread
    Ls = ntuple(i->deepcopy(L), nthreads())
    Js = ntuple(i->deepcopy(J), nthreads())

    # temporary
    tmp = similar(z[1])

    # period and shift
    if NS == 2
        T, s = z.d
    else
        T,   = z.d
    end

    # get last point on the orbit and obtain its time derivative 
     x0   =   store(similar(z[1]), 0, Val(0))
     xT   = NS == 2 ? S(store(similar(z[1]), T, Val(0)), s) : store(similar(z[1]), T, Val(0))
    dxTdT = NS == 2 ? S(store(similar(z[1]), T, Val(1)), s) : store(similar(z[1]), T, Val(1))

    # right hand side
    rhs = similar(z)

    @threads :static for i = 1:N
        # note that store must be thread safe
        id = threadid()

        # integration span
        span = ((i-1)*T/N, i*T/N)

        # set homogeneous initial condition
        rhs[i] .= 0

        # integrate non-homogeneous equations over the i-th span
        Js[id](rhs[i], store, span)

        # flip sign for the right hand side
        rhs[i] .*= -1.0
    end

    # we need to "back-shift" the last state it if we have a symmetry
    NS == 2 && S(rhs[N], s)

    rhs.d = tuple(zeros(NS)...)

    return TangentProblemLHS(Ls, S, D, x0, xT, dxTdT, tmp, z, store), rhs
end

# outer constructor without shift
make_tangent_problem(z::MVector{X, N, 1},
                     store,
                     L,
                     J,
                     D) where {X, N} = make_tangent_problem(z,
                                                            store,
                                                            L,
                                                            J,
                                                            nothing,
                                                            D)

# Main interface is matrix-vector product exposed to the Krylov solver
Base.:*(mm::TangentProblemLHS{X}, w::MVector{X}) where {X} = mul!(similar(w), mm, w)

# Compute mat-vec product
function mul!(out::MVector{X, N, NS},
              mm::TangentProblemLHS{X, N, NS},
              w::MVector{X, N, NS}) where {X, N, NS}
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

    # compute L{x0[i]}⋅w[i] - w[i+1]
    @threads :static for i = 1:N
        # this thread id
        id = threadid()

        # set perturbation initial condition
        out[i] .= w[i]

        # integration span
        span = ((i-1)*T/N, i*T/N)

        # integrate linearised equations
        Ls[id](out[i], store, span)

        # apply shift on last segment
        NS == 2 && i == N && S(out[i], s)

        # this is the identity operators on the upper diagonal
        out[i] .-= w[i%N + 1]
    end

    # Add time and space derivative of state at end point. Note that
    # if we have a spatial continuous symmetry (NS = 2), these two 
    # derivatives will have been "back-shifted"
    out[N] .+= dxTdT.*w.d[1]
    NS == 2 && (out[N] .+= D[2](tmp, xT).*w.d[2])

    # Compute phase locking constraints
    out.d = ntuple(j->dot(w[1], D[j](tmp, x0)), length(D))

    return out
end
