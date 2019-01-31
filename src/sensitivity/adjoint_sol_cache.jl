# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import Base.Threads: threadid, @threads, nthreads
import LinearAlgebra: dot
import Flows

export make_adjoint_problem

# ~~~ Matrix Type ~~~
struct AdjointProblemLHS{X, N, NS, M, LST, SType, DType, CS}
        Ls::LST               # homogeneous adjoint operators (one per thread)
         S::SType             # space shift operator
         D::DType             # time (and space) derivative operator
    dxTdTs::NTuple{N, X}      # time derivative of flow operator
       xTs::NTuple{N, X}      # final points
      tmps::NTuple{M, X}      # temporary storage
         z::MVector{X, N, NS} # the periodic orbit
    caches::CS                # the stage caches for the adjoint integration
end

# Main outer constructor
function make_adjoint_problem(G,
                              L,
                              J,
                              S,
                              D,
                              z::MVector{X, N, NS},
                          cache::Flows.AbstractStageCache,
                              ϵ::Real) where {X, N, NS}
                        
    # Construct temporaries. The last element of tmps 
    # is used to store the (shifted) final state. 
    tmps = ntuple(i->similar(z[1]), 2*nthreads())

    # this stores the derivative of the discrete flow operator
    dxTdTs = similar.(z.x)
     xTs   = similar.(z.x)

    # monitors to store the state before the last step is made
    mons = ntuple(i->Flows.StoreOneButLast(tmps[1]), nthreads())

    # this will be the right hand side
    rhs = similar(z)

    # store all caches here
    caches = ntuple(i->similar(cache), N)

    # the period and shift
    T, s = z.d

    # make copies of the propagators
    Gs = ntuple(i->deepcopy(G), nthreads())
    Js = ntuple(i->deepcopy(J), nthreads())

    # loop over all segments
    @threads for i = 1:N
        # this thread id
        id = threadid()

        # set initial condition
        xTs[i] .= z[i]

        # propagate filling the cache and storing 
        # the state before the last step
        Gs[id](xTs[i], (0, T/N), caches[i], mons[id])

        # Calculate derivative of the flow operator wrt T using
        # centered finite difference. This just requires little
        # computations, because we only propagate from the state
        # before the last step is done
        tmps[2*id] .= mons[id].x; tmps[2*id-1] .= mons[id].x;
        Gs[id](tmps[2*id],   (mons[id].t, T/N + ϵ))
        Gs[id](tmps[2*id-1], (mons[id].t, T/N - ϵ))
        dxTdTs[i] .= 0.5.*(tmps[2*id] .- tmps[2*id-1])./ϵ
    end

    # construct right hand side by propagating an homogeneous
    # adjoint state backwards with the adjoint state transition
    # operator that includes the forcing term
    @threads for i = 1:N
        id = threadid()
        rhs[i] .= 0
        Js[id](rhs[i], caches[i])
        rhs[i] .*= -1.0
    end

    # we keep the last bits to zero too
    rhs.d = zero.(rhs.d)

    # last elements need shifting
    NS == 2 && (S(xTs[N], s); S(dxTdTs[N], s))

    # construct object
    AdjointProblemLHS(ntuple(i->deepcopy(L), nthreads()), S, D, 
                      dxTdTs, xTs, tmps, z, caches), rhs
end


# Main interface is matrix-vector product exposed to the Krylov solver
Base.:*(A::AdjointProblemLHS{X}, w::MVector{X}) where {X} = mul!(similar(w), A, w)

# Compute mat-vec product (version including one spatial shifts)
function mul!(out::MVector{X, N, 2},
                A::AdjointProblemLHS{X, N, 2},
                w::MVector{X, N, 2}) where {X, N}
    # aliases
    dxTdTs = A.dxTdTs
    caches = A.caches
    tmps   = A.tmps
    xTs    = A.xTs
    Ls     = A.Ls
    D      = A.D
    S      = A.S
    z      = A.z
    T, s   = z.d

    # main block
    @threads for i = 1:N
        out[i] .= w[i]
        i == N && S(out[i], -s)
        Ls[threadid()](out[i], caches[i])
        out[i] .-= w[(i+N-2)%N + 1]
    end

    # right column
    out[1] .+= D[1](tmps[1], z[1]).*w.d[1] .+ D[2](tmps[2], z[1]).*w.d[2]

    # bottom row
    out.d = (sum(dot(w[j], dxTdTs[j]) for j = 1:N)/N, dot(D[2](tmps[1], xTs[N]), w[N]))

    return out
end


# Compute mat-vec product (version for no shifts)
function mul!(out::MVector{X, N, 1},
                A::AdjointProblemLHS{X, N, 1},
                w::MVector{X, N, 1}) where {X, N}
    # aliases
    xT    = A.xT
    Ls    = A.Ls
    D     = A.D
    z     = A.z
    tmps   = A.tmps
    dxTdTs = A.dxTdTs

    # main block
    @threads for i = 1:N
        out[i] .= w[i]
        Ls[threadid()](out[i], caches[i])
        out[i] .-= w[(i+N-2)%N + 1]
    end

    # right column
    out[1] .+= D[1](tmps, z[1]).*w.d[1]

    # bottom row
    out.d = (sum(dot(w[j], dxTdTs[j]) for j = 1:N)/N, )

    return out
end