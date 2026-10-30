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

# the rotation of (x, y) by s, its inverse (what the adjoint problem takes), and its generator
rpo_shift(X, s) = (σ = first(s); x = X[1]; y = X[2]; X[1] = cos(σ)*x - sin(σ)*y; X[2] = sin(σ)*x + cos(σ)*y; X)
rpo_unshift(X, s) = rpo_shift(X, -first(s))
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
    A, b = make_adjoint_problem(z, store, LA, JA, rpo_unshift, D, jTJ)
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

    # ---- the same shift, split differently between the closure and the unknown ----
    # S_off(X, σ) = R(σ + θ₀) with θ₀ fixed and the shift unknown σ = s - θ₀. Every FORWARD
    # application is unchanged, R(σ + θ₀) = R(s), so the orbit still closes, but the inverse is
    # R(-(σ + θ₀)), not R(-σ): negating the unknown alone does not invert this closure. That is what
    # a shift carrying a symmetry outside the unknowns looks like — in Kolmogorov flow the continuous
    # x-shift is bundled with a fixed discrete y-shift. rpo_generator! is still the right D[2], since
    # ∂_σ S_off(x, σ) is evaluated at u(0) and the extra rotation commutes with the one-parameter
    # group, and dT/dμ, dσ/dμ are the same numbers because θ₀ does not depend on μ.
    #
    # The split is a pure relabelling, so both assemblies must be unchanged by it, to round-off.
    # Compare the operators and the right-hand sides, not only what the solves return: this system's
    # adjoint is driven by a functional of (u, v) alone and carries almost no weight in the rotated
    # (x, y) plane, so the solution is a far weaker probe than the mat-vec.
    θ₀        = 0.7
    S_off     = (X, σ) -> rpo_shift(X,   first(σ) + θ₀)
    S_inv_off = (X, σ) -> rpo_shift(X, -(first(σ) + θ₀))
    z_off     = MVector(copy.(z.x), T, s - θ₀)

    # both problem builders deep-copy these per segment, so one pair serves every assembly below
    Lo = flow(RPOTangent{false}(), RK4(zeros(4), Flows.ContinuousMode(false)), TimeStepFromStorage(dt))
    Jo = flow(RPOTangent{true}(),  RK4(zeros(4), Flows.ContinuousMode(false)), TimeStepFromStorage(dt))

    # a yardstick for the comparisons below
    scale(a)     = sqrt(sum(norm(a[i])^2 for i in 1:N) + sum(abs2, a.d))
    differ(a, b) = sqrt(sum(norm(a[i] - b[i])^2 for i in 1:N) + sum(abs2, a.d .- b.d))

    for (Ap, bp, Ao, bo) in ((NKSearch.make_tangent_problem(z,     store, Lo, Jo, rpo_shift,   D)...,
                              NKSearch.make_tangent_problem(z_off, store, Lo, Jo, S_off,       D)...),
                             (make_adjoint_problem(z,     store, LA, JA, rpo_unshift, D, jTJ)...,
                              make_adjoint_problem(z_off, store, LA, JA, S_inv_off,   D, jTJ)...))
        @test differ(bp, bo) < 1e-12*scale(bp)
        for trial = 1:3
            v = MVector(ntuple(i -> randn(4), N), randn(), randn())
            @test differ(Ap*v, Ao*v) < 1e-12*scale(Ap*v)
        end
    end

    # and the solves, end to end
    Ao, bo = NKSearch.make_tangent_problem(z_off, store, Lo, Jo, S_off, D)
    wo, _  = NKSearch.gmres!(zero(bo), Ao, bo; rel_rtol=1e-12, maxiter=50, verbose=false)
    @test abs(wo.d[1] - Tp) < 1e-6*abs(Tp)
    @test abs(wo.d[2] - sp) < 1e-6*abs(sp)

    Ao, bo = make_adjoint_problem(z_off, store, LA, JA, S_inv_off, D, jTJ)
    λo, _  = NKSearch.gmres!(zero(bo), Ao, bo; rel_rtol=1e-12, maxiter=50, verbose=false)
    @test differ(λo, λ) < 1e-8*scale(λ)

    # ---- the adjoint problem by the stage caches gives the same solution ----
    LA = flow(RPOAdjoint{false}(), RK4(zeros(4), Flows.DiscreteMode(true)), TimeStepFromCache())
    JA = flow(RPOAdjoint{true}(),  RK4(zeros(4), Flows.DiscreteMode(true)), TimeStepFromCache())
    A, b = make_adjoint_problem(z, cache, LA, JA, rpo_unshift, D, jTJ)
    λc, res = NKSearch.gmres!(zero(b), A, b; rel_rtol=1e-12, maxiter=50, verbose=false)
    δ = sqrt(sum(norm(λc[i] - λ[i])^2 for i in 1:N) + sum(abs2, λc.d .- λ.d))
    @test δ < 1e-6*sqrt(sum(norm(λ[i])^2 for i in 1:N) + sum(abs2, λ.d))

    # ---- the adjoint operator is the exact transpose of the tangent operator ----
    # The adjoint unknown is a relabelling of the multipliers of the tangent rows: component m
    # multiplies tangent row perm[m], the multiplier of the wrap-around row is held in the frame of
    # u(T) (pre-multiplied by Sᵀ) and the scalar unknowns are negated, the sign convention of the
    # `-=` on the border row r. The assembly also leaves its own output on row r pre-multiplied by
    # Sᵀ, so one forward shift puts it back. Undo all of that and ⟨Av, y⟩ = ⟨v, Aᵀy⟩ must hold.
    #
    # The stage caches are used because they make the segment operators exact discrete transposes of
    # each other; the store path is only transpose-accurate to the integration error. Both shift
    # parameterisations are run: the offset one is what catches an assembly that goes back to
    # deriving its inverse by negating the unknown.
    Lc  = flow(RPOTangent{false}(), RK4(zeros(4), Flows.DiscreteMode(false)), TimeStepFromCache())
    Jc  = flow(RPOTangent{true}(),  RK4(zeros(4), Flows.DiscreteMode(false)), TimeStepFromCache())
    LAc = flow(RPOAdjoint{false}(), RK4(zeros(4), Flows.DiscreteMode(true)),  TimeStepFromCache())
    JAc = flow(RPOAdjoint{true}(),  RK4(zeros(4), Flows.DiscreteMode(true)),  TimeStepFromCache())

    for (zq, Sf, Si) in ((z, rpo_shift, rpo_unshift), (z_off, S_off, S_inv_off)),
        row_order in (:ashtari, :regular)

        Atan, _ = NKSearch.make_tangent_problem(zq, cache, Lc, Jc, Sf, D; row_order=row_order)
        Aadj, _ = make_adjoint_problem(zq, cache, LAc, JAc, Si, D, jTJ; row_order=row_order)

        # adjoint component m multiplies tangent row perm[m]; perm is an involution and perm[1] is
        # the border row r, the one the assembly shifts (N = 2 here)
        perm = row_order == :ashtari ? ntuple(m -> m == 1 ? 1 : N - m + 2, N) : ntuple(m -> N - m + 1, N)
        σ    = Base.tail(zq.d)

        for trial = 1:3
            v  = MVector(ntuple(i -> randn(4), N), randn(), randn())
            q  = MVector(ntuple(i -> randn(4), N), randn(), randn())
            Av = Atan*v
            Aq = Aadj*q

            # the tangent-row multipliers the adjoint unknown q stands for
            y = ntuple(i -> zeros(4), N)
            y[perm[1]] .= Sf(copy(q[1]), σ)
            for m = 2:N
                y[perm[m]] .= q[m]
            end

            # and the adjoint output, relabelled back and un-shifted, which must be Aᵀy
            ŷ = ntuple(p -> p == 1 ? Sf(copy(Aq[perm[1]]), σ) : Aq[perm[p]], N)

            lhs = sum(dot(Av[p], y[p]) for p = 1:N) - sum(Av.d .* q.d)
            rhs = sum(dot(v[p],  ŷ[p]) for p = 1:N) + sum(v.d  .* Aq.d)
            @test abs(lhs - rhs) < 1e-9*max(abs(lhs), abs(rhs))
        end
    end
end
