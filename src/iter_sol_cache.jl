# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import Base.Threads: threadid, @threads, nthreads
import LinearAlgebra: dot
import GMRES: gmres!
import Flows

# ~~~ Matrix Type ~~~
struct IterSolCache{X, N, NS, M, GST, LST, ST, DT, MT}
       Gs::GST               # flow operator(s)
       Ls::LST               # linearised flow operator (s)
        S::ST                # space shift operator
        D::DT                # time (and space) derivative operators
       xT::NTuple{N, X}      # time shifted conditions
    dxTdT::NTuple{N, X}      # time derivative of flow operator
      tmp::NTuple{M, X}      # temporary storage
       z0::MVector{X, N, NS} # current orbit
     mons::MT                # monitor
     opts::Options           # options
end

# Main outer constructor
IterSolCache(Gs, Ls, S, D, z0::MVector{X, N, NS}, opts) where {X, N, NS} =
    IterSolCache(Gs, Ls, S, D,
                 similar.(z0.x),
                 similar.(z0.x),
                 ntuple(i->similar(z0[1]), 2*nthreads() + 2),
                 similar(z0),
                 ntuple(i->Flows.StoreOneButLast(z0[1]), nthreads()),
                 opts)

# Main interface is matrix-vector product exposed to the Krylov solver
Base.:*(mm::IterSolCache{X}, δz::MVector{X}) where {X} = mul!(similar(δz), mm, δz)

# Compute mat-vec product
function mul!(out::MVector{X, N, NS},
               mm::IterSolCache{X, N, NS},
               δz::MVector{X, N, NS}) where {X, N, NS}
    # aliases
    xT    = mm.xT
    Ls    = mm.Ls
    D     = mm.D
    S     = mm.S
    z0    = mm.z0
    tmp   = mm.tmp
    dxTdT = mm.dxTdT
    T     = mm.z0.d[1]
    ϵ     = mm.opts.ϵ

    # compute L{x0[i]}⋅δz[i] - δz[i+1]
    @threads :static for i = 1:N
        # this thread id
        id = threadid()

        # set perturbation initial condition
        out[i] .= δz[i]

        # set nonlinear initial condition
        tmp[id] .= z0[i]

        # propagate by T/N
        Ls[id](Flows.couple(tmp[id], out[i]), (0, T/N))

        # apply shift on last segment (if we have one)
        NS == 2 && i == N && S(out[i], z0.d[2])

        # this is the identity operators on the upper diagonal
        out[i] .-= δz[i%N + 1]
    end

    # period derivative
    for i = 1:N
        out[i] .+= dxTdT[i].*(δz.d[1]./N)
    end

    # shift derivative (if present) goes only on last element
    NS == 2 && (out[N] .+= D[2](tmp[1], xT[N]).*δz.d[2])

    # add phase locking constraints
    out.d = ntuple(j->dot(δz[1], D[j](tmp[1], z0[1])), NS)

    return out
end

# Update the linear operator and rhs arising in the Newton-Raphson iterations
function update!(mm::IterSolCache{X, N, NS},
                  b::MVector{X, N, NS},
                 z0::MVector{X, N, NS},
               opts::Options) where {X, N, NS}

    # store this vector for the products
    mm.z0 .= z0

    # aliases
    xT    = mm.xT
    Gs    = mm.Gs
    S     = mm.S
    tmp   = mm.tmp
    dxTdT = mm.dxTdT
    ϵ     = opts.ϵ
    T     = z0.d[1]
    mons  = mm.mons

    # update initial and final states
    @threads :static for i = 1:N
        # this thread ID
        id = threadid()

        # set and propagate
        xT[i] .= z0[i]
        Gs[id](xT[i], (0, T/N), mons[id])

        # finite difference derivative of flow operator
        # see https://epubs.siam.org/doi/10.1137/070705623 page 27
        tmp[2*id  ] .= mons[id].x;
        tmp[2*id-1] .= mons[id].x;
        Gs[id](tmp[2*id  ], (mons[id].t, T/N + ϵ))
        # ! this can cause a bug since mons[id].t < T/N - ϵ
        Gs[id](tmp[2*id-1], (mons[id].t, T/N - ϵ))
        dxTdT[i] .= 0.5.*(tmp[2*id] .- tmp[2*id-1])./ϵ
    end

    # last one (may) get shifted
    NS == 2 && S(   xT[N], z0.d[2])
    NS == 2 && S(dxTdT[N], z0.d[2])

    # ~~ RIGHT HAND SIDE ~~
    # calculate negative error
    for i = 1:N
        b[i] .= z0[i%N+1] .- xT[i]
    end

    # reset shifts
    b.d = zero.(b.d)

    return nothing
end

# solution for iterative method
_solve(x::MV, A::IterSolCache, b::MV, opts::Options) where {MV<:MVector} =
    gmres!(x, A, b; rel_rtol=opts.gmres_rtol,
                 maxiter=opts.gmres_maxiter,
                 verbose=opts.gmres_verbose,
                 trace=opts.gmres_trace)

_solve(x::MV, A::IterSolCache, b::MVector, tr_radius::Real, opts::Options) where {MV<:MVector} =
    gmres!(x, A, b, tr_radius; rel_rtol=opts.gmres_rtol,
                            maxiter=opts.gmres_maxiter,
                            verbose=opts.gmres_verbose,
                            trace=opts.gmres_trace)
