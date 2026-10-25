# Eigenvalues of the NKSearch Newton operator A (Ashtari and regular ordering)
# and of the right-preconditioned operator A M⁻¹, assembled column by column,
# for KS along a chaotic trajectory: N = 32 segments of length τ = 2.
# Writes data/spectrum.txt with columns: re, im, label.
include(joinpath(@__DIR__, "ks.jl"))
using NKSearch, Random, LinearAlgebra

const τ, N, k = 2.0, 32, 8
G, L, F = ks_flows()
Random.seed!(1)
a = 0.1*randn(32); G(a, (0, 300.0))
seeds = [copy(a)]
for i in 2:N
    G(a, (0, τ)); push!(seeds, copy(a))
end
z = MVector(Tuple(seeds), N*τ)

function assemble(op, z)
    n = length(NKSearch.tovector(z))
    M = zeros(n, n); e = zero(z)
    for j in 1:n
        v = zeros(n); v[j] = 1
        NKSearch.fromvector!(e, v)
        M[:, j] .= NKSearch.tovector(op*e)
    end
    return M
end

io = open(joinpath(@__DIR__, "data", "spectrum.txt"), "w")
for (label, order, kk) in (("regular", :regular, 0), ("ashtari", :ashtari, 0), ("precond", :ashtari, k))
    opts = Options(method=:ls_iterative, row_order=order, coarse_dim=kk, gmres_verbose=false)
    A = NKSearch.IterSolCache(ntuple(i->deepcopy(G), N), ntuple(i->deepcopy(L), N), nothing, (F,), z, opts)
    NKSearch.update!(A, similar(z), z, opts)
    op = kk == 0 ? A : NKSearch.RightPrecOp(A, A.pc, similar(z))
    λ = eigvals(assemble(op, z))
    foreach(l->println(io, real(l), " ", imag(l), " ", label), λ)
    println(label, ": min |λ| = ", minimum(abs, λ), ", max |λ| = ", maximum(abs, λ))
end
close(io)
