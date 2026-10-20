# ----------------------------------------------------------------- #
# Test: spatial shift (NS=2) — relative periodic orbits              #
# ----------------------------------------------------------------- #
# The Hopf normal form is rotationally symmetric, so the exact
# solution is the unit circle with period 2π and spatial shift 0.
#
# This file verifies that the 7-argument (L-BFGS with adjoint)
# search! signature correctly handles the spatial shift operators
# S, F, dS, and that the residual decreases.

# Spatial shift: rotate state x by angle s (in place)
struct SpatialShift end
function (::SpatialShift)(x, s)
    c, sn = cos(s), sin(s)
    x1, x2 = x[1], x[2]
    x[1] = c*x1 - sn*x2
    x[2] = sn*x1 + c*x2
    return x
end

# Generator of the spatial shift: d/ds S(x, s) evaluated at s=0
struct SpatialShiftDerivative end
function (::SpatialShiftDerivative)(out, x)
    out[1] = -x[2]
    out[2] =  x[1]
    return out
end

# =========================================================================
#  Smoke test: verify the L-BFGS + spatial-shift loop runs and the
#  residual decreases significantly in the first few iterations.
# =========================================================================
@testset "search_lbfgs_shift smoke (NS=2)        " begin
    μ = 1.0
    F_sys = System(μ)
    D     = SystemLinear(μ)
    D_adj = SystemLinearAdjoint(μ)

    S_op  = SpatialShift()
    dS_op = SpatialShiftDerivative()

    F_phase = (out, x) -> F_sys(0, x, out)

    G = flow(F_sys,
             RK4(zeros(2), Flows.NormalMode()),
             TimeStepConstant(1e-3))
    L = flow(TangentSystem(D),
             RK4(zeros(2), Flows.DiscreteMode(false)),
             TimeStepFromCache())
    adj_flow = flow(AdjointTangentSystem(D_adj),
                    RK4(zeros(2), Flows.DiscreteMode(true)),
                    TimeStepFromCache())

    z = MVector(([1.1, 0.0], [-1.1, 0.0]), 2π, 0.0)

    residuals = Float64[]
    cb = (iter, z, Fz, e_norm, ∇ϕ_norm, λ, T) -> push!(residuals, e_norm)

    status = search!(G, L, adj_flow, S_op, F_phase, dS_op, z,
                     Options(maxiter=20,
                             dz_norm_tol=0.0,
                             e_norm_tol=0.0,
                             verbose=true,
                             method=:lbfgs_opt,
                             ls_maxiter=20,
                             lbfgs_memory=20,
                             callback=cb))

    @test length(residuals) >= 2                     # at least iter 0 + 1
    @test residuals[end] < residuals[1] * 0.5        # F-norm at least halved
    @test status == :maxiter_reached                 # stopped by maxiter, not crashed
end

# =========================================================================
#  Newton–Krylov searches of a relative periodic orbit with two segments.
#  In cylindrical coordinates (ρ, θ, x₃) the system has the Hopf limit
#  cycle of the Viswanath problem in the plane (ρ - 2, x₃), of period 2π,
#  and rotates at the constant rate Ω around the x₃ axis, so that the orbit
#  closes up to the rotation by -2πΩ. The rotation of (x₁, x₂) is the
#  spatial shift. With more than one segment, the derivative with respect
#  to the shift belongs to the row of the closure of the last segment,
#  row 1 with the ordering :ashtari and row N with :regular.
# =========================================================================
struct RotatingHopf
    μ::Float64
    Ω::Float64
end
function (s::RotatingHopf)(t, x, dxdt)
    ρ = sqrt(x[1]^2 + x[2]^2)
    a = ρ - 2
    b = x[3]
    q = 1 - (a^2 + b^2)
    ρ̇ = -b + s.μ*a*q
    dxdt[1] = ρ̇/ρ*x[1] - s.Ω*x[2]
    dxdt[2] = ρ̇/ρ*x[2] + s.Ω*x[1]
    dxdt[3] =  a + s.μ*b*q
    return dxdt
end

struct RotationXY end
function (::RotationXY)(x, s)
    c, sn = cos(s), sin(s)
    x1, x2 = x[1], x[2]
    x[1] = c*x1 - sn*x2
    x[2] = sn*x1 + c*x2
    return x
end

struct RotationXYGenerator end
function (::RotationXYGenerator)(out, x)
    out[1] = -x[2]
    out[2] =  x[1]
    out[3] =  0.0
    return out
end

@testset "search_tr_iterative_shift (NS=2, N=2) " begin
    μ, Ω = 1.0, 0.3
    F_sys = RotatingHopf(μ, Ω)
    G     = flow(F_sys, RK4(zeros(3), Flows.NormalMode()), TimeStepConstant(1e-3))
    L     = JFOp(G, zeros(3), 1e-7)
    F     = (out, x) -> F_sys(0, x, out)
    sexact = rem(-2π*Ω, 2π, RoundNearest)

    # the orbit at θ = 0, x₃ = 0 and half a period later, perturbed: Newton converges in a few
    # iterations with either ordering (with the shift derivative in the wrong row, the trust-region
    # search took 25 linear iterations with :ashtari)
    for row_order in (:ashtari, :regular)
        z = MVector(([3.05, 0.0, 0.02], [cos(π*Ω)*0.98, sin(π*Ω)*0.98, -0.03]), 2π + 0.05, sexact + 0.04)
        iters = Ref(0)
        status = search!(G, L, RotationXY(), F, RotationXYGenerator(), z,
                         Options(method=:tr_iterative, row_order=row_order, maxiter=30, e_norm_tol=1e-10,
                                 dz_norm_tol=1e-14, min_step=1e-14, gmres_maxiter=20, gmres_rtol=1e-12,
                                 verbose=false, gmres_verbose=false,
                                 callback=(args...) -> (iters[] += 1; false)))
        @test status == :converged
        @test iters[] <= 5
        @test abs(z.d[1] - 2π) < 1e-6
        @test abs(rem(z.d[2] - sexact, 2π, RoundNearest)) < 1e-6
    end
end
