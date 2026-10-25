# First Lyapunov exponents of KS (L = 39, odd subspace) by QR iteration over segments of length 2.
include(joinpath(@__DIR__, "ks.jl"))
using Random, LinearAlgebra, Printf
Random.seed!(1)
G, L, F = ks_flows()
a = 0.1*randn(32); G(a, (0, 300.0))
Q = Matrix(qr(randn(32, 32)).Q); λ = zeros(32); x = copy(a)
for n in 1:300
    for j in 1:32
        xb = copy(x); q = Q[:, j]; L(couple(xb, q), (0, 2.0)); Q[:, j] .= q
    end
    G(x, (0, 2.0))
    F_ = qr(Q); λ .+= log.(abs.(diag(F_.R))); Q .= Matrix(F_.Q)
end
λ ./= 600
for j in 1:14
    @printf "%2d  λ = %7.3f   e^(λτ) = %.3f\n" j λ[j] exp(2λ[j])
end
