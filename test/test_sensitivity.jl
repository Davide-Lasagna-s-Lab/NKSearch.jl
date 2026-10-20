# ----------------------------------------------------------------- #
# Test: tangent and adjoint sensitivity of a relative periodic orbit #
# ----------------------------------------------------------------- #
# A relative periodic orbit known in closed form, with a period and a shift that depend on the parameter μ. The state
# is (x, y, u, v). The pair (u, v) is a Hopf oscillator with a limit cycle of radius √μ and frequency Ω = ω + βμ; the
# point (x, y) relaxes to the unit circle and turns at the rate c + κ(u² + v²). The rotation of (x, y) is the
# continuous symmetry. On the orbit
#
#     T = 2π/Ω ,   s = -(c + κμ) T   (the shift that closes the orbit, S(G(x₀, T), s) = x₀) ,
#
# and the mean of u² is μ/2. The tangent problem must return dT/dμ and ds/dμ, and the adjoint problem the
# derivative 1/2 of the mean of u², through both the RAMStorage and the RAMStageCache paths. The shift makes the
# end point of the orbit differ from its start, S(u(T), s) = u(0), which the periodic orbits of the other tests do
# not exercise.

const μ_rpo = 1.0
const ω_rpo = 1.0
const β_rpo = 0.5
const c_rpo = 0.3
const κ_rpo = 0.2

# the right-hand side
function rpo_rhs!(dX, X)
    x, y, u, v = X
    R2 = u^2 + v^2
    r2 = x^2 + y^2
    dX[1] = (1 - r2)*x - (c_rpo + κ_rpo*R2)*y
    dX[2] = (1 - r2)*y + (c_rpo + κ_rpo*R2)*x
    dX[3] = (μ_rpo - R2)*u - (ω_rpo + β_rpo*R2)*v
    dX[4] = (μ_rpo - R2)*v + (ω_rpo + β_rpo*R2)*u
    return dX
end

# its Jacobian
function rpo_jacobian(X)
    x, y, u, v = X
    R2 = u^2 + v^2
    r2 = x^2 + y^2
    a  = c_rpo + κ_rpo*R2
    b  = ω_rpo + β_rpo*R2
    return [1 - r2 - 2x^2   -2x*y - a        -2κ_rpo*u*y                      -2κ_rpo*v*y;
            -2x*y + a       1 - r2 - 2y^2     2κ_rpo*u*x                       2κ_rpo*v*x;
            0.0             0.0               μ_rpo - R2 - 2u^2 - 2β_rpo*u*v   -2u*v - b - 2β_rpo*v^2;
            0.0             0.0              -2u*v + b + 2β_rpo*u^2            μ_rpo - R2 - 2v^2 + 2β_rpo*u*v]
end

# the systems, as named callable structs (see runtests.jl): the flow, the tangent with and without the forcing
# ∂F/∂μ = (0, 0, u, v), and the adjoint with and without the forcing -∂J/∂X of J = u²
struct RPOSystem end
(::RPOSystem)(t, X, dX) = rpo_rhs!(dX, X)

struct RPOTangent{F} end
function (::RPOTangent{F})(t, X, V, dV) where {F}
    mul!(dV, rpo_jacobian(X), V)
    F && (dV[3] += X[3]; dV[4] += X[4])
    return dV
end

struct RPOAdjoint{F} end
function (::RPOAdjoint{F})(t, X, W, dW) where {F}
    mul!(dW, rpo_jacobian(X)', W)
    F && (dW[3] -= 2X[3])
    return dW
end

# the rotation of (x, y) by s, and its generator
rpo_shift(X, s) = (σ = first(s); x = X[1]; y = X[2]; X[1] = cos(σ)*x - sin(σ)*y; X[2] = sin(σ)*x + cos(σ)*y; X)
rpo_generator!(out, X) = (out .= (-X[2], X[1], 0.0, 0.0); out)

@testset "sensitivity of a relative periodic orbit   " begin
    # ---- the orbit in closed form, on two segments, and its exact derivatives in μ ----
    Ω  = ω_rpo + β_rpo*μ_rpo
    T  = 2π/Ω
    s  = -(c_rpo + κ_rpo*μ_rpo)*T
    Tp = -2π*β_rpo/Ω^2
    sp = -κ_rpo*T + (c_rpo + κ_rpo*μ_rpo)*2π*β_rpo/Ω^2
    N  = 2
    θ  = (c_rpo + κ_rpo*μ_rpo)*T/N
    z  = MVector(([1.0, 0.0, sqrt(μ_rpo), 0.0], [cos(θ), sin(θ), sqrt(μ_rpo)*cos(Ω*T/N), sqrt(μ_rpo)*sin(Ω*T/N)]), T, s)
    D  = ((out, X) -> rpo_rhs!(out, X), rpo_generator!)

    # ---- the flow, a store that is not periodic and ends at u(T) unshifted, and the stage caches ----
    dt    = 1e-3
    G     = flow(RPOSystem(), RK4(zeros(4), Flows.NormalMode()), TimeStepConstant(dt))
    store = RAMStorage(Vector{Float64}; storelast=false)
    cache = ntuple(i -> RAMStageCache(4, zeros(4)), N)
    for i = 1:N
        G(copy(z[i]), ((i - 1)*T/N, i*T/N), store)
        G(copy(z[i]), ((i - 1)*T/N, i*T/N), cache[i])
    end
    push!(store, T, G(copy(z[N]), ((N - 1)*T/N, T)))

    # ---- the tangent problem: dT/dμ and ds/dμ, by both paths ----
    for (data, mode, stepping) in ((store, Flows.ContinuousMode(false), TimeStepFromStorage(dt)),
                                   (cache, Flows.DiscreteMode(false),   TimeStepFromCache()))
        L = flow(RPOTangent{false}(), RK4(zeros(4), mode), stepping)
        J = flow(RPOTangent{true}(),  RK4(zeros(4), mode), stepping)
        A, b = NKSearch.make_tangent_problem(z, data, L, J, rpo_shift, D)
        w, res = NKSearch.gmres!(zero(b), A, b; rel_rtol=1e-12, maxiter=50, verbose=false)
        @test abs(w.d[1] - Tp) < 1e-6*abs(Tp)
        @test abs(w.d[2] - sp) < 1e-6*abs(sp)
    end

    # ---- the adjoint problem by the store: the derivative of the mean of u², -(1/T)∫⟨∂F/∂μ, λ⟩ dt = 1/2 ----
    LA  = flow(RPOAdjoint{false}(), RK4(zeros(4), Flows.ContinuousMode(true)), TimeStepFromStorage(dt))
    JA  = flow(RPOAdjoint{true}(),  RK4(zeros(4), Flows.ContinuousMode(true)), TimeStepFromStorage(dt))
    jTJ = z[1][3]^2 - μ_rpo/2
    A, b = make_adjoint_problem(z, store, LA, JA, rpo_shift, D, jTJ)
    λ, res = NKSearch.gmres!(zero(b), A, b; rel_rtol=1e-12, maxiter=50, verbose=false)

    # (λ along each segment, backwards from its value at the start of the segment, and the integral by the trapezoids)
    integral = 0.0
    X = zeros(4)
    for i = 1:N
        mon = Monitor(zeros(4), (t, w) -> copy(w))
        JA(copy(λ[i]), store, (T - (i - 1)*T/N, T - i*T/N), mon)
        ts = times(mon)
        fs = [(store(X, t); X[3]*samples(mon)[k][3] + X[4]*samples(mon)[k][4]) for (k, t) in enumerate(ts)]
        integral += sum(abs(ts[k + 1] - ts[k])*(fs[k] + fs[k + 1])/2 for k in 1:length(ts) - 1)
    end
    @test abs(-integral/T - 0.5) < 1e-5

    # ---- the adjoint problem by the stage caches gives the same solution ----
    LA = flow(RPOAdjoint{false}(), RK4(zeros(4), Flows.DiscreteMode(true)), TimeStepFromCache())
    JA = flow(RPOAdjoint{true}(),  RK4(zeros(4), Flows.DiscreteMode(true)), TimeStepFromCache())
    A, b = make_adjoint_problem(z, cache, LA, JA, rpo_shift, D, jTJ)
    λc, res = NKSearch.gmres!(zero(b), A, b; rel_rtol=1e-12, maxiter=50, verbose=false)
    δ = sqrt(sum(norm(λc[i] - λ[i])^2 for i in 1:N) + sum(abs2, λc.d .- λ.d))
    @test δ < 1e-6*sqrt(sum(norm(λ[i])^2 for i in 1:N) + sum(abs2, λ.d))
end
