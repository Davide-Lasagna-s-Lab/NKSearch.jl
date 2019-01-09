# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

import Flows

# ~~~ Matrix Type ~~~
struct MMatrix{X, N, NS, GType, LType, SType, DType}
        G::GType             # flow operator with no shifts
        L::LType             # linearised flow operator with no shifts
        S::SType             # space shift operator (can be NoShift)
        D::DType             # time (and space) derivative operator
       xT::NTuple{N, X}      # time shifted conditions
    dxTdT::NTuple{N, X}      # time derivative of flow operator
       z0::MVector{X, N, NS} # current orbit
      tmp::X                 # temporary storage
     opts::Options           # 
end

# Main outer constructor
MMatrix(G, L, S, D, z0::MVector{X, N, NS}, opts) where {X, N, NS} =
    MMatrix(G, L, S, D, similar.(z0.x), similar.(z0.x), z0, similar(z0[1]), opts)

# Main interface is matrix-vector product exposed to the Krylov solver
Base.:*(mm::MMatrix{X}, δz::MVector{X}) where {X} = mul!(similar(δz), mm, δz)

# Compute mat-vec product
function mul!(out::MVector{X, N, NS},
               mm::MMatrix{X, N, NS},
               δz::MVector{X, N, NS}) where {X, N, NS}
    # aliases
    xT    = mm.xT
    G     = mm.G
    L     = mm.L
    D     = mm.D
    S     = mm.S
    z0    = mm.z0
    tmp   = mm.tmp
    dxTdT = mm.dxTdT
    T     = mm.z0.d[1]
    ϵ     = mm.opts.ϵ

    # compute L{x0[i]}⋅δz[i] - δz[i+1]
    for i = 1:N
        # set perturbation initial condition
        out[i] .= δz[i]

        # set nonlinear initial condition
        tmp      .= z0[i]

        # propagate by T/N
        L(Flows.couple(tmp, out[i]), (0, T/N))

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
    NS == 2 && (out[N] .+= D[2](tmp, xT[N]).*δz.d[2])

    # add phase locking constraints
    out.d = ntuple(j->dot(δz[1], D[j](tmp, z0[1])), NS)

    return out
end

# Update the linear operator and rhs arising in the Newton-Raphson iterations
function update!(mm::MMatrix{X, N, NS},
                  b::MVector{X, N, NS},
               opts::Options) where {X, N, NS}

    # aliases
    xT    = mm.xT
    G     = mm.G
    S     = mm.S
    z0    = mm.z0
    tmp   = mm.tmp
    dxTdT = mm.dxTdT
    ϵ     = opts.ϵ
    T     = mm.z0.d[1]

    # update initial and final states
    for i = 1:N
        # set and propagate
        xT[i] .= z0[i]
        G(xT[i], (0, T/N))
        
        # then get finite difference approximation of derivative
        # of the flow operator with one additional propagation
        tmp .= z0[i]
        G(tmp, (0, T/N + ϵ))
        dxTdT[i] .= (tmp .- xT[i])./ϵ

        # last one gets shifted (if we have one)
        NS == 2 && i == N && S(   xT[i], z0.d[2])
        NS == 2 && i == N && S(dxTdT[i], z0.d[2])
    end

    # ~~ RIGHT HAND SIDE ~~
    # calculate negative error
    for i = 1:N
        b[i] .= z0[i%N+1] .- xT[i]
    end

    # reset shifts
    b.ds = zero.(b.ds)

    return nothing
end