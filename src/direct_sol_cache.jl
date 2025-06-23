# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import Base.Threads: threadid, @threads, nthreads
import SparseArrays: spzeros, SparseMatrixCSC, diagind
import LinearAlgebra: lu, ldiv!
import Flows: couple


# Apply operator op{u}*v to every column v of the identity matrix
# ! this function doesn't work if a complex valued array type is passed
function op_apply_eye!(out::Matrix{T},
                       op,
                       u::X,
                       span::Tuple{Real, Real},
                       _u::X, _v::X) where {T, X}
    n = length(u)
    @inbounds for i = 1:n
        _u    .= u
        _v    .= 0
        _v[i]  = 1
        out[:, i] .= op(_u, _v, span)
    end
    return out
end

struct DirectSolCache{GST, LST, ST, DT, YST, TMPST, MONST}
      Gs::GST               # flow operator with no shifts
      Ls::LST               # linearised flow operator with no shifts
       S::ST                # space shift operator (can be NoShift)
       D::DT                # time (and space) derivative operator
       A::SparseMatrixCSC{Float64, Int}
      Ys::YST               # temporaries
    tmps::TMPST
    mons::MONST             # monitor
end

function DirectSolCache(Gs, Ls, S, D, z0::MVector{X, N, NS}, opts) where {X, N, NS}
    n = length(z0[1])
    m = N*n + NS
    DirectSolCache(Gs,
                   Ls,
                   S,
                   D,
                   spzeros(m, m),
                   ntuple(i->zeros(n, n), nthreads()),
                   ntuple(i->similar(z0[1]), 3*nthreads()), 
                   ntuple(i->Flows.StoreOnlyLast(z0[1]), nthreads()))
end

Base.:*(dsm::DirectSolCache, dz::MVector) = fromvector!(similar(dz), dsm.A*tovector(dz))

function update!(dsm::DirectSolCache,
                   b::MVector{X, N, NS},
                  z0::MVector{X, N, NS},
                opts::Options) where {X, N, NS}
    # length of vector
    n = length(b[1])

    # aliases
       A = dsm.A; A.nzval .= 0
    mons = dsm.mons
    tmps = dsm.tmps
      Ys = dsm.Ys
      Ls = dsm.Ls
      Gs = dsm.Gs
       S = dsm.S
       D = dsm.D
       T = z0.d[1]
       s = NS == 2 ? z0.d[2] : 0.0

    # linear operators on the diagonal
    op1(id) = (x, y, span)->(Ls[id](couple(x, y), span); y)
    op2(id) = (x, y, span)->(Ls[id](couple(x, y), span); S(y, s); y)

    # fill main block
    # @threads 
    for i = 1:N
        # this thread id
        id = 1#threadid()
        
        # global indices where we will write
        rng = _blockrng(i, n)

        # write i-th block on diagonal
        if NS == 2 && i == N
            op_apply_eye!(Ys[id], op2(id), z0[i], (0, T/N), tmps[3*id], tmps[3*id-1])
        else
            # ! this passes FTField objects when it needs to be passing vectors
            op_apply_eye!(Ys[id], op1(id), z0[i], (0, T/N), tmps[3*id], tmps[3*id-1])
        end
        A[rng, rng] .= Ys[id]
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
    A[end - NS + 1, _blockrng(1, n)] .= D[1](tmps[1], z0[1])
    if NS == 2
        A[end, _blockrng(1, n)] .= D[2](tmps[1], z0[1])
    end

    # right borders and right hand side
    # @threads 
    for i = 1:N
        # id of current thread
        id = 1#threadid()

        # aliases
        _u1 = tmps[3*id]
        _u2 = tmps[3*id-1]
        _u3 = tmps[3*id-2]

        # obtain final state
        _u3 .= z0[i]
        Gs[id](_u3, (0, T/N), mons[id])

        # calc derivative of flow operator
        _u1 .= mons[id].x
        _u2 .= mons[id].x;
        Gs[id](_u1, (mons[id].t, T/N + opts.ϵ))
        Gs[id](_u2, (mons[id].t, T/N - opts.ϵ))
        _u2 .= 0.5.*(_u1 .- _u2)./opts.ϵ

        #apply shift if needed
        NS == 2 && i == N && S(_u2, s)

        # write along column
        A[_blockrng(i, n), end - NS + 1] .= _u2./N

        # write column for the shift
        if NS == 2 && i == N
            S(_u3, s)
            D[2](_u2, _u3)
            A[_blockrng(i, n), end] .= _u2
        end

        # rhs
        b[i] .= z0[i % N + 1] .- _u3
    end

    # reset shifts
    b.d = zero.(b.d)

    return nothing
end


# solution for direct method
_solve(dz::MVector, dsm::DirectSolCache, b::MVector, opts::Options) =
    (fromvector!(dz, ldiv!(lu(dsm.A), tovector(b))), 0.0)
