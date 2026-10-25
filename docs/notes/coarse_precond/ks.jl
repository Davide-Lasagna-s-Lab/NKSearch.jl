# Kuramoto-Sivashinsky u_t = -u u_x - u_xx - u_xxxx on [0, L) in the odd
# subspace u(-x) = -u(x), u = Σ_{k=1}^{K} a_k sin(κ_k x), κ_k = 2πk/L.
# The state is the vector of the K sine coefficients. The linear part is
# diagonal and treated implicitly; the advection term is evaluated on a grid of
# NX points (NX ≥ 3K + 1 removes aliasing of the quadratic term).
using Flows, LinearAlgebra

struct KS
    κ::Vector{Float64}
    S::Matrix{Float64}      # NX × K, sine synthesis
    C::Matrix{Float64}      # K × NX, cosine analysis
    u::Vector{Float64}      # grid buffers
    v::Vector{Float64}
    w::Vector{Float64}
    c::Vector{Float64}      # coefficient buffer
end

function KS(L::Float64, K::Int, NX::Int)
    κ = [2π*k/L for k in 1:K]
    x = [L*j/NX for j in 0:NX-1]
    S = [sin(κ[k]*x[j]) for j in 1:NX, k in 1:K]
    C = [2/NX*cos(κ[k]*x[j]) for k in 1:K, j in 1:NX]
    KS(κ, S, C, zeros(NX), zeros(NX), zeros(NX), zeros(K))
end

# linear operator -∂xx - ∂xxxx in the sine basis
linear(ks::KS) = Diagonal(ks.κ.^2 .- ks.κ.^4)

# nonlinear term -(u²/2)_x: u² is even, its cosine coefficients c_k give -(c_k cos)'/2 = κ_k c_k/2 sin
function (ks::KS)(t, a, dadt)
    mul!(ks.u, ks.S, a)
    ks.w .= ks.u.^2
    mul!(ks.c, ks.C, ks.w)
    dadt .= 0.5.*ks.κ.*ks.c
    return dadt
end

# linearised nonlinear term -(u v)_x
struct KSTangent
    ks::KS
end

function (kt::KSTangent)(t, a, dadt, b, dbdt)
    ks = kt.ks
    mul!(ks.u, ks.S, a)
    mul!(ks.v, ks.S, b)
    ks.w .= ks.u.*ks.v
    mul!(ks.c, ks.C, ks.w)
    dbdt .= ks.κ.*ks.c
    return dbdt
end

# right-hand side of the full equation, F(out, a), for the phase condition
struct KSRhs
    ks::KS
    A::Diagonal{Float64, Vector{Float64}}
end
(f::KSRhs)(out, a) = (f.ks(0.0, a, out); out .+= f.A.diag.*a; out)

# Nonlinear flow G, coupled tangent flow L and right-hand side F
function ks_flows(; L=39.0, K=32, NX=128, dt=0.025)
    ks = KS(L, K, NX)
    A = linear(ks)
    G = flow(ks, A, CB3R2R3e(zeros(K)), TimeStepConstant(dt))
    Lt = flow(couple(ks, KSTangent(ks)), couple(A, A),
              CB3R2R3e(couple(zeros(K), zeros(K))), TimeStepConstant(dt))
    return G, Lt, KSRhs(ks, A)
end
