# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import Flows

# ~~~ Matrix Type ~~~
mutable struct MMatrix{X, N, GType, LType, SType, dGType, dSType}
      G::GType              # flow operator with no shifts
      L::LType              # linearised flow operator with no shifts
      S::SType              # space shift operator
     dG::dGType             # derivative of G wrt to time
     dS::dSType             # derivative of S wrt to the shift
     x0::NTuple{N, X}       # current initial seeds
     xT::NTuple{N, X}       # time shifted conditions
      T::NTuple{3, Float64} # orbit period
      s::Float64            # spatial shift
    tmp::X                  # temporary storage
end

# Main outer constructor
MMatrix(G, L, S, dG, dS, z0::MVector{X, N}) where {X, N} =
    MMatrix(G, L, S, dG, dS,
             ntuple(j->similar(z0[1]), N), ntuple(j->similar(z0[1]), N),
             z0.T, z0.s, similar(z0[1]))

# Main interface is matrix-vector product exposed to the Krylov solver
Base.:*(A::MMatrix{X}, δz::MVector{X}) where {X} = A_mul_B!(similar(δz), A, δz)

# Compute mat-vec product
function Base.A_mul_B!(out::MVector{X, N}, 
                         A::MMatrix{X, N}, 
                        δz::MVector{X, N}) where {X, N}
    # aliases
    x0, xT, T, s, δT, δs, tmp = A.x0, A.xT, A.T, A.s, δz.T, δz.s, A.tmp
    G, L, S, dG, dS = A.G, A.L, A.S, A.dG, A.dS

    # compute L{x0[i]}⋅δz[i] - δz[i+1] (last element gets shifted)
    for i = 1:N
        out[i] .= δz[i] # set perturbation initial condition
        tmp    .= x0[i] # set nonlinear    initial condition
        # use the correct propagation time
        Ti = i == 1 ? T[1] : i == N ? T[3] : T[2]/(N-2)
        L(Flows.couple(tmp, out[i]), (0, Ti))
        i == N && S(out[i], s)
        out[i] .-= δz[i%N+1]
    end

    # period derivative
    for i = 1:N
        d = i == 1 ? δT[1] : i == N ? δT[3] : δT[2]/(N-2)
        out[i] .+= dG(tmp, xT[i]).*d
    end

    # shift derivative
    out[N] .+= dS(tmp, xT[N]).*δs
   
    # phase locking constraints
    a     = dot(δz[1], dG(tmp, x0[1]))
    b     = dot(δz[2], dG(tmp, x0[2]))
    c     = dot(δz[N], dG(tmp, x0[N]))
    out.T = (a, b, c)
    out.s = dot(δz[1], dS(tmp, x0[1]))

    return out
end

# Update the linear operator and rhs arising in the Newton-Raphson iterations
function update!(A::MMatrix{X, N},
                 b::MVector{X, N},
                z0::MVector{X, N}) where {X, N}
    # update shifts
    A.T, A.s, b.T, b.s = z0.T, z0.s, (0.0, 0.0, 0.0), 0.0
    
    # aliases
    x0, xT, G, S, T, s = A.x0, A.xT, A.G, A.S, A.T, A.s
    
    # update initial and final states
    for i = 1:N
        x0[i] .= z0[i]
        xT[i] .= x0[i]
        Ti = i == 1 ? T[1] : i == N ? T[3] : T[2]/(N-2)
        # do not care about the right time span for autonomous systems
        G(xT[i], (0, Ti))
        # last one gets shifted
        i == N && S(xT[i], s)
    end

    # calculate negative error
    for i = 1:N
        b[i] .= x0[i%N+1] .- xT[i]
    end

    return nothing
end