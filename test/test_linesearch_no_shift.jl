# Problem from section 4 in Viswanath 2001
# ẋ = -y + μx(1 - √(x² + y²))
# ẏ =  x + μy(1 - √(x² + y²))
# The solution is the limit cycle
# x(t) = cos(t)
# y(t) = sin(t)
# with period T=2π and angular frequency ω=1
# The velocities are:
# ẋ(t) = -sin(t)
# ẏ(t) =  cos(t)

struct System
    μ::Float64
end

# right hand side
function (s::System)(t, x, dxdt)
    x_, y_, μ = x[1], x[2], s.μ
    r = sqrt(x_^2 + y_^2)
    @inbounds dxdt[1] = - y_ + μ*x_*(1 - r)
    @inbounds dxdt[2] =   x_ + μ*y_*(1 - r)
    return dxdt
end

# linearised operator
struct SystemLinear
    μ::Float64
    J::Matrix{Float64}
    SystemLinear(μ::Real) = new(μ, zeros(2, 2))
end

function (s::SystemLinear)(t, x, dxdt, v, dvdt)
    x_, y_, μ = x[1], x[2], s.μ
    r = sqrt(x_^2 + y_^2)
    @inbounds s.J[1, 1] = μ*(1 - r - x_^2/r)
    @inbounds s.J[1, 2] = -1 - μ*x_*y_/r
    @inbounds s.J[2, 1] =  1 - μ*x_*y_/r
    @inbounds s.J[2, 2] = μ*(1 - r - y_^2/r)
    return mul!(dvdt, s.J, v)
end

@testset "search_linesearch                      " begin
    # define systems
    μ = 1.0
    F = System(μ)
    D = SystemLinear(μ)

    # define propagators
    G = flow(F,
             RK4(zeros(2), :NORMAL),
             TimeStepConstant(1e-3))
    L = flow(couple(F, D),
             RK4(couple(zeros(2), zeros(2)), :NORMAL),
             TimeStepConstant(1e-3))

    # for method in (:linesearch,)
        # for solver in (:direct, :iterative)
    for method in (:hookstep,)
        for solver in (:direct,)
            # define initial guess, a slightly perturbed orbit
            z = MVector(([2, 0.0], [-2, 0.0]), 2π)

            # search
            search!(G,
                    L,
                    (dxdt, x)->F(0, x, dxdt),
                    z,
                    Options(maxiter=15,
                            dz_norm_tol=1e-16,
                            gmres_verbose=false,
                            e_norm_tol=1e-16,
                            verbose=true,
                            tr_radius_init=1,
                            method=method,
                            solver=solver))

            # solution is a loop of unit radius and with T = 2π
            # @test maximum( map(el->norm(el)-1, z.x) ) < 1e-10
            # @test abs(z.d[1] - 2π ) < 1e-10
        end
    end
end