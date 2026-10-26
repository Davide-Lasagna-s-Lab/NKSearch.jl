# ----------------------------------------------------------------- #
# Coarse-space right preconditioner for the Ashtari-ordered          #
# multiple-shooting Newton operator.                                 #
# ----------------------------------------------------------------- #
import SparseArrays: sparse, SparseMatrixCSC
import SparseArrays.SPQR: QRSparse
import Random: randn!

# With the Ashtari ordering the Newton operator is A = -I + C (plus the period
# and shift columns and the phase rows), where C holds the segment tangent maps
# M_i on the block subdiagonal (and S∘M_N in the corner). For every Floquet
# multiplier μ, the eigenvalues of -I + C are -1 + μ^(1/N) e^(2πik/N): a circle
# of radius |μ|^(1/N) = e^(λτ) centred at -1. Circles of contracting directions
# stay away from the origin and GMRES converges at a rate set by τ only. Circles
# of the unstable and neutral directions surround the origin, and GMRES needs
# about N iterations for them. We remove them with a coarse space: at each seed
# i, an orthonormal basis V_i of the leading covariant subspace, of dimension k.
# These subspaces are invariant, M_i span(V_i) = span(V_{i+1}), so the Galerkin
# coarse problem acts on them exactly.
#
# Coarse unknowns: c_i ∈ R^k for every seed, plus the NS scalars (period, shift).
# W = blockdiag(V_1, …, V_N, I_NS) is orthonormal and E = Wᵀ A W. The right
# preconditioner is
#
#           M⁻¹ = -(I - W Wᵀ) + W E⁻¹ Wᵀ,     with inverse   M = -(I - W Wᵀ) + W E Wᵀ,
#
# i.e. -I outside the coarse space (the inverse of the Ashtari diagonal) and an
# exact coarse solve on it.
#
# The bases are obtained by subspace iteration around the orbit, all segments in
# parallel: V_{i+1} ← orth(M_i V_i). The leading covariant subspace at a point
# depends only on the recent past, so the number of rounds needed does not grow
# with N. The first Newton step runs `coarse_sweeps` rounds from random bases;
# each later step reuses the images of the previous step as a warm start, so its
# only cost is one round: k tangent integrations per segment.
mutable struct CoarsePrecond{X, N, NS}
    V::Vector{Vector{X}}               # V[i][a]: a-th basis vector at seed i
    P::Vector{Vector{X}}               # P[i][a]: image of V[i][a] over segment i (shifted if i = N)
    base::Vector{X}                    # one base-flow buffer per segment
    tmp::X                             # buffer for derivative operators
    E::SparseMatrixCSC{Float64, Int}   # coarse matrix
    F::QRSparse{Float64, Int}          # its QR factorisation (LU with pivoting is unstable here)
    g::Vector{Float64}                 # coarse buffers
    y::Vector{Float64}
    initialised::Bool

    function CoarsePrecond{X, N, NS}(z0::MVector{X, N, NS}, k::Int) where {X, N, NS}
        V = [[similar(z0[1]) for a in 1:k] for i in 1:N]
        P = [[similar(z0[1]) for a in 1:k] for i in 1:N]
        n = N*k + NS
        E = sparse(1.0*I(n))
        new{X, N, NS}(V, P, [similar(z0[1]) for i in 1:N], similar(z0[1]),
                      E, qr(E), zeros(n), zeros(n), false)
    end
end

coarse_dim(pc::CoarsePrecond) = length(pc.V[1])

# range of the coarse unknowns of seed i
_crng(i::Int, k::Int) = ((i-1)*k+1):(i*k)

# orthonormalise in place with modified Gram-Schmidt, applied twice
function _mgs!(Q::Vector{X}) where {X}
    for _ in 1:2, a in eachindex(Q)
        for b in 1:a-1
            Q[a] .-= dot(Q[b], Q[a]).*Q[b]
        end
        Q[a] ./= norm(Q[a])
    end
    return Q
end

# P_i ← M_i V_i on all segments in parallel; the last segment is shifted
function _images!(pc::CoarsePrecond{X, N, NS}, Ls, S, z0::MVector{X, N, NS}) where {X, N, NS}
    T = z0.d[1]
    @sync for i in 1:N
        @spawn for a in eachindex(pc.V[i])
            pc.P[i][a] .= pc.V[i][a]
            pc.base[i] .= z0[i]
            Ls[i](Flows.couple(pc.base[i], pc.P[i][a]), (0, T/N))
            NS == 2 && i == N && S(pc.P[i][a], z0.d[2])
        end
    end
    return pc
end

# one step of subspace iteration: V_{i+1} ← orth(P_i)
function _advance!(pc::CoarsePrecond{X, N}) where {X, N}
    for i in 1:N
        V = pc.V[i%N + 1]
        for a in eachindex(V)
            V[a] .= pc.P[i][a]
        end
        _mgs!(V)
    end
    return pc
end

# Galerkin coarse matrix E = Wᵀ A W for the Ashtari ordering
function _assemble!(pc::CoarsePrecond{X, N, NS}, D, z0, xT, dxTdT) where {X, N, NS}
    k = coarse_dim(pc)
    Is, Js, Vs = Int[], Int[], Float64[]
    entry!(r, c, v) = (push!(Is, r); push!(Js, c); push!(Vs, v))
    for i in 1:N
        j = i%N + 1                                     # row block of segment i
        for a in 1:k, b in 1:k
            entry!(_crng(j, k)[a], _crng(i, k)[b], dot(pc.V[j][a], pc.P[i][b]))
        end
        for a in 1:k
            entry!(_crng(j, k)[a], _crng(j, k)[a], -1.0)
            entry!(_crng(j, k)[a], N*k + 1, dot(pc.V[j][a], dxTdT[i])/N)
        end
    end
    if NS == 2
        D[2](pc.tmp, xT[N])
        for a in 1:k
            entry!(_crng(1, k)[a], N*k + 2, dot(pc.V[1][a], pc.tmp))
        end
    end
    for d in 1:NS
        D[d](pc.tmp, z0[1])
        for b in 1:k
            entry!(N*k + d, _crng(1, k)[b], dot(pc.tmp, pc.V[1][b]))
        end
    end
    n = N*k + NS
    pc.E = sparse(Is, Js, Vs, n, n)
    pc.F = qr(pc.E)
    return pc
end

# Build the coarse space and the coarse matrix at the current orbit z0. Must be
# called after xT and dxTdT have been computed (and shifted) by `update!`.
function update!(pc::CoarsePrecond{X, N, NS}, Ls, S, D, z0, xT, dxTdT, opts::Options) where {X, N, NS}
    if pc.initialised
        _advance!(pc)
    else
        for i in 1:N
            foreach(randn!, pc.V[i])
            _mgs!(pc.V[i])
        end
        for _ in 1:opts.coarse_sweeps
            _images!(pc, Ls, S, z0)
            _advance!(pc)
        end
        pc.initialised = true
    end
    _images!(pc, Ls, S, z0)
    _assemble!(pc, D, z0, xT, dxTdT)
    return pc
end

# g ← Wᵀ r
function _restrict!(g::Vector{Float64}, pc::CoarsePrecond{X, N, NS}, r::MVector{X, N, NS}) where {X, N, NS}
    k = coarse_dim(pc)
    for i in 1:N, a in 1:k
        g[_crng(i, k)[a]] = dot(pc.V[i][a], r[i])
    end
    for d in 1:NS
        g[N*k + d] = r.d[d]
    end
    return g
end

# out ← -(I - W Wᵀ) r + W y, with g = Wᵀ r
function _prolong!(out::MVector{X, N, NS}, pc::CoarsePrecond{X, N, NS}, r::MVector{X, N, NS},
                   g::Vector{Float64}, y::Vector{Float64}) where {X, N, NS}
    k = coarse_dim(pc)
    for i in 1:N
        out[i] .= .-r[i]
        for a in 1:k
            out[i] .+= (g[_crng(i, k)[a]] + y[_crng(i, k)[a]]).*pc.V[i][a]
        end
    end
    out.d = ntuple(d->y[N*k + d], NS)
    return out
end

# out ← M⁻¹ r
function ldiv_precond!(out::MVector, pc::CoarsePrecond, r::MVector)
    _restrict!(pc.g, pc, r)
    pc.y .= pc.F \ pc.g
    return _prolong!(out, pc, r, pc.g, pc.y)
end

# out ← M r, used to map a warm start into the preconditioned variable
function mul_precond!(out::MVector, pc::CoarsePrecond, r::MVector)
    _restrict!(pc.g, pc, r)
    pc.y .= pc.E*pc.g
    return _prolong!(out, pc, r, pc.g, pc.y)
end

# A M⁻¹, the operator GMRES sees
struct RightPrecOp{C, PC, MV}
    A::C
    pc::PC
    tmp::MV
end

Base.:*(op::RightPrecOp, y::MVector) = op.A*ldiv_precond!(op.tmp, op.pc, y)

# no coarse preconditioner
update!(pc::Nothing, args...) = nothing
