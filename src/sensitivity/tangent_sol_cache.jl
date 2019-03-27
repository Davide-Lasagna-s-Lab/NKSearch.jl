# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import Base.Threads: threadid, @threads, nthreads
import LinearAlgebra: dot
import GMRES: gmres!
import Flows

# ~~~ Matrix Type ~~~
struct TangentProblemLHS{X, N, NS, M, LST, ST, DT, MT}
       Ls::LST               # linearised flow operator(s)
        S::ST                # space shift operator
        D::DT                # time (and space) derivative operators
      xTs::NTuple{N, X}      # time shifted conditions
   dxTdTs::NTuple{N, X}      # time derivative of flow operator
     tmps::NTuple{M, X}      # temporary storage
        z::MVector{X, N, NS} # current orbit
     mons::MT                # monitors
end

# Main outer constructor
function make_tangent_problem(G,
                              L,
                              J,
                              S,
                              D,
                              z::MVector{X, N, NS},
                              ϵ::Real=1e-6) where {X, N, NS}

    # make copies of the propagators, one for each thread
    Gs = ntuple(i->deepcopy(G), 1)#nthreads())
    Ls = ntuple(i->deepcopy(L), 1)#nthreads())
    Js = ntuple(i->deepcopy(J), 1)#nthreads())

    # this stores the derivative of the discrete flow operator
    dxTdTs = similar.(z.x)

    # and the final states
    xTs    = similar.(z.x)

    # Construct temporaries
    tmps = ntuple(i->similar(z[1]), 2)#2*nthreads())

    # monitors to store the state before the last step is made
    mons = ntuple(i->Flows.StoreOneButLast(tmps[1]), 1)#nthreads())

    # the period and shift
    T, s = z.d

    # this will be the right hand side
    rhs = similar(z)

    # update initial and final states
    # @threads 
    for i = 1:N
        # this thread ID
        id = 1#threadid()

        # set and propagate
        xTs[i] .= z[i]
        Gs[id](xTs[i], (0, T/N), mons[id])

        # finite difference derivative of flow operator
        tmps[2*id  ] .= mons[id].x;
        tmps[2*id-1] .= mons[id].x;
        Gs[id](tmps[2*id  ], (mons[id].t, T/N + ϵ))
        Gs[id](tmps[2*id-1], (mons[id].t, T/N - ϵ))
        dxTdTs[i] .= 0.5.*(tmps[2*id] .- tmps[2*id-1])./ϵ
    end

    # last one (may) get shifted
    NS == 2 && S(   xTs[N], s)
    NS == 2 && S(dxTdTs[N], s)

    # ~~ RIGHT HAND SIDE ~~
    # construct right hand side by propagating an homogeneous
    # tangent state forward with the tangent state transition
    # operator that includes the forcing term
    rhs .*= 0
    # @threads 
    for i = 1:N
        id = 1#threadid()
        tmps[id] .= z[i]
        Js[id](Flows.couple(tmps[id], rhs[i]), (0, T/N))
        rhs[i] .*= -1.0
    end
    NS == 2 && S(rhs[N], s)

    return TangentProblemLHS(Ls, S, D, xTs, dxTdTs, tmps, z, mons), rhs
end

# Main interface is matrix-vector product exposed to the Krylov solver
Base.:*(mm::TangentProblemLHS{X}, w::MVector{X}) where {X} = mul!(similar(w), mm, w)

# Compute mat-vec product
function mul!(out::MVector{X, N, 2},
               mm::TangentProblemLHS{X, N, 2},
                w::MVector{X, N, 2}) where {X, N}

    # aliases
    xTs    = mm.xTs
    Ls     = mm.Ls
    D      = mm.D
    S      = mm.S
    z      = mm.z
    tmps   = mm.tmps
    dxTdTs = mm.dxTdTs
    T, s   = mm.z.d

    # compute L{x0[i]}⋅w[i] - w[i+1]
    @threads for i = 1:N
        # this thread id
        id = threadid()

        # set perturbation initial condition
        out[i] .= w[i]

        # set nonlinear initial condition
        tmps[id] .= z[i]

        # propagate by T/N
        Ls[id](Flows.couple(tmps[id], out[i]), (0, T/N))

        # apply shift on last segment
        i == N && S(out[i], s)

        # this is the identity operators on the upper diagonal
        out[i] .-= w[i%N + 1]
    end

    # period derivative
    for i = 1:N
        out[i] .+= dxTdTs[i].*(w.d[1]./N)
    end

    # shift derivative
    out[N] .+= D[2](tmps[1], xTs[N]).*w.d[2]

    # add phase locking constraints
    out.d = ntuple(j->dot(w[1], D[j](tmps[1], z[1])), 2)

    return out
end
