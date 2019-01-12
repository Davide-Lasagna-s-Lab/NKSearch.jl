# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

import LinearAlgebra: lu, ldiv!
import SparseArrays: spzeros, SparseMatrixCSC, diagind
import Flows: couple


# Apply operator op{u}*v to every column v of the identity matrix
function op_apply_eye!(out::Matrix{T},
                       op,
                       u::X,
                       span::Tuple{Real, Real},
                       _u::X, _v::X) where {T, X<:AbstractVector{T}}
    n = length(u)
    @inbounds for i = 1:n
        _u    .= u
        _v    .= 0
        _v[i]  = 1
        out[:, i] .= op(couple(_u, _v), span)[2]
    end
    return out
end

struct DirectSolCache{X, GType, LType, SType, DType}
        G::GType             # flow operator with no shifts
        L::LType             # linearised flow operator with no shifts
        S::SType             # space shift operator (can be NoShift)
        D::DType             # time (and space) derivative operator
        A::SparseMatrixCSC{Float64, Int}
        Y::Matrix{Float64}   # temporaries
      tmp::NTuple{2, X}
end

function DirectSolCache(G, L, S, D, z0::MVector{X, N, NS}, opts) where {X, N, NS}
    n = length(z0[1])
    m = N *n + NS
    DirectSolCache(G,
                   L,
                   S,
                   D,
                   spzeros(m, m),
                   zeros(n, n),
                   ntuple(i->similar(z0.x[1]), 2))
end

function update!(dsm::DirectSolCache,
                   b::MVector{X, N, NS},
                  z0::MVector{X, N, NS},
                opts::Options) where {X, N, NS}
    # length of vector
    n = length(b[1])

    # aliases
     A = dsm.A
     Y = dsm.Y
    _u = dsm.tmp[1]
    _v = dsm.tmp[2]
     L = dsm.L
     D = dsm.D
     G = dsm.G
     T = z0.d[1]

    # fill main block
    for i = 1:N
        rng = _blockrng(i, n)
        A[rng, rng] .= op_apply_eye!(Y, L, z0[i], (0, T/N), _u, _v)
    end

    # add identities on diagonals
    A[diagind(A,        n)] .= -1
    A[diagind(A, -(N-1)*n)] .= -1

    # reset borders
    A[end, :] .= 0
    A[:, end] .= 0

    # lower borders
    A[end, _blockrng(1, n)] .= D[1](_u, z0[1])

    # right borders and right hand side
    for i = 1:N
        _u .= z0[i]
        G(_u, (0, T/N))
        A[_blockrng(i, n), end] .= D[1](_v, _u)./N
        b[i] .= z0[i % N + 1] .- _u
    end

    # reset shifts
    b.d = zero.(b.d)

    return nothing
end


# solution for direct method
_solve(dsm::DirectSolCache, b::MVector, opts::Options) =
    (fromarray!(b, ldiv!(lu(dsm.A), toarray(b))), 0.0)
