# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import LinearAlgebra: dot
import GMRES: gmres!
import Flows
using Base.Threads

# ~~~ Matrix Type ~~~
mutable struct IterSolCache{X, N, NS, M, GType, LType, SType, DType}
       Gs::Vector{GType}                                  # flow operator with no shifts
       Ls::Vector{LType}                                  # linearised flow operator with no shifts
        S::SType                                          # space shift operator (can be NoShift)
        D::DType                                          # time (and space) derivative operator
       xT::NTuple{N, X}                                   # time shifted conditions
    dxTdT::NTuple{N, X}                                   # time derivative of flow operator
       z0::MVector{X, N, NS}                              # current orbit
     mons::Vector{Flows.StoreOneButLast{X, typeof(copy)}} # monitor
      tmp::NTuple{M, X}                                   # temporary storage
     opts::Options                                        #
end

# Main outer constructor
IterSolCache(G, L, S, D, z0::MVector{X, N, NS}, opts) where {X, N, NS} =
    IterSolCache([deepcopy(G) for i = 1:N], 
                 [deepcopy(L) for i = 1:N], S, D, similar.(z0.x), similar.(z0.x), 
                 similar(z0), [Flows.StoreOneButLast(z0[1]) for i = 1:N], 
                 ntuple(i->similar(z0[1]), 2N + 2), opts)

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
    Threads.@threads for i = 1:N
        # set perturbation initial condition
        out[i] .= δz[i]

        # set nonlinear initial condition
        tmp[i] .= z0[i]

        # propagate by T/N
        Ls[i](Flows.couple(tmp[i], out[i]), (0, T/N))

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
    Threads.@threads for i = 1:N
        # set and propagate
        xT[i] .= z0[i]
        Gs[i](xT[i], (0, T/N), mons[i])

        # finite difference derivative of flow operator 
        tmp[i] .= mons[i].x; tmp[2i] .= mons[i].x;
        Gs[i](tmp[ i], (mons[i].t, T/N + ϵ))
        Gs[i](tmp[2i], (mons[i].t, T/N - ϵ))
        dxTdT[i] .= 0.5.*(tmp[i] .- tmp[2i])./ϵ
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
_solve(A::IterSolCache, b::MVector, opts::Options) =
    gmres!(A, b; rtol=opts.gmres_rtol,
                 maxiter=opts.gmres_maxiter,
                 verbose=opts.gmres_verbose)

_solve(A::IterSolCache, b::MVector, tr_radius::Real, opts::Options) =
    gmres!(A, b, tr_radius; rtol=opts.gmres_rtol,
                            maxiter=opts.gmres_maxiter,
                            verbose=opts.gmres_verbose)