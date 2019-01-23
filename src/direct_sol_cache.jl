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
        out[:, i] .= op(_u, _v, span)
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
      tmp::NTuple{3, X}
      mon::Flows.StoreOneButLast{X, typeof(copy)} # monitor
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
                   ntuple(i->similar(z0.x[1]), 3), 
                   Flows.StoreOneButLast(z0.x[1]))
end

Base.:*(dsm::DirectSolCache{X}, dz::MVector{X}) where {X} = 
    fromvector!(similar(dz), dsm.A*tovector(dz))

function update!(dsm::DirectSolCache,
                   b::MVector{X, N, NS},
                  z0::MVector{X, N, NS},
                opts::Options) where {X, N, NS}
    # length of vector
    n = length(b[1])

    # aliases
     A = dsm.A; A.nzval .= 0
   mon = dsm.mon
     Y = dsm.Y
    _u = dsm.tmp[1]
    _v = dsm.tmp[2]
    _w = dsm.tmp[3]
     S = dsm.S
     L = dsm.L
     D = dsm.D
     G = dsm.G
     T = z0.d[1]
     s = NS == 2 ? z0.d[2] : 0.0

    op1(x, y, span) = (L(couple(x, y), span); y)
    op2(x, y, span) = (L(couple(x, y), span); S(y, s); y)

    # fill main block
    for i = 1:N
        rng = _blockrng(i, n)
        if NS == 2 && i == N
            op_apply_eye!(Y, op2, z0[i], (0, T/N), _u, _v)
        else
            op_apply_eye!(Y, op1, z0[i], (0, T/N), _u, _v)
        end
        A[rng, rng] .= Y
    end

    # add identities on diagonals
    if N == 1
        A[diagind(A,        0)] .+= -1
    else
        A[diagind(A,        n)] .+= -1
        A[diagind(A, -(N-1)*n)] .+= -1
    end

    # reset borders
    A[end - NS + 1:end, :] .= 0
    A[:, end - NS + 1:end] .= 0

    # lower borders
    A[end - NS + 1, _blockrng(1, n)] .= D[1](_u, z0[1])
    if NS == 2
        A[end, _blockrng(1, n)] .= D[2](_u, z0[1])
    end

    # right borders and right hand side
    for i = 1:N
        _w .= z0[i]
        G(_w, (0, T/N), mon)

        _u .= mon.x; _v .= mon.x;
        G(_u, (mon.t, T/N + opts.ϵ))
        G(_v, (mon.t, T/N - opts.ϵ))
        _v .= 0.5.*(_u .- _v)./opts.ϵ

        NS == 2 && i == N && S(_v, s)

        A[_blockrng(i, n), end - NS + 1] .= _v./N

        if NS == 2 && i == N
            S(_w, s)
            D[2](_v, _w)
            A[_blockrng(i, n), end] .= _v
        end

        b[i] .= z0[i % N + 1] .- _w
    end

    # reset shifts
    b.d = zero.(b.d)

    return nothing
end


# solution for direct method
_solve(dsm::DirectSolCache, b::MVector, opts::Options) =
    (fromvector!(b, ldiv!(lu(dsm.A), tovector(b))), 0.0)
