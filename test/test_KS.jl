using Base.Test
using Flows, KS, asis

@testset "test f and g! for single shooting      " begin
    # parameters
    ν        = (2π/22)^2
    Δt       = 0.1*ν
    ISODD    = false
    n        = 10

    # seed rng
    srand(0)

    # system right hand side
    F = ForwardEquation(n, ν, ISODD)

    # integrator
    ϕ = flow(splitexim(F)..., 
                   CB3R2R3e(FTField(n, ISODD), :NORMAL), TimeStepConstant(Δt))

    # land on attractor
    U0 = ϕ(FTField(n, ISODD, k->exp(2π*im*rand())/k), (0, 200*ν))

    # define initial conditions
    T    = 200*ν
    TL   = 50*ν
    TR   = 50*ν    
    U0_C = ϕ(copy(U0), (0, TL)) 
    UF_C = ϕ(copy(U0), (0, T-TR)) 

    # rhs
    L⁺ = LinearisedEquation(n, ν, ISODD, KS.AdjointMode())

    # integrators
    ψ⁺ = flow(splitexim(L⁺)..., 
                   CB3R2R3e(FTField(n, ISODD), :ADJ), TimeStepFromCache())

    # define the stage cache
    caches = (RAMStageCache(4, FTField(n, ISODD)), RAMStageCache(4, FTField(n, ISODD)))

    # build cache for gradient evaluation
    f, g! = GradientCache(ϕ, F, KS.ddx!, ψ⁺, U0_C, UF_C, caches)

    # initial condition
    x0 = tovector(copy(U0), copy(UF_C), TL, TR, 0)

    # preallocate and calc gradient
    grad_exact = zeros(x0)
    g!(grad_exact, x0)

    # //// TEST GRADIENT WITH RESPECT TO COMPONENTS ////
    eps = 1e-6
    for k = 1:2n
        s = eps*abs(grad_exact[k])
        x0[k] -= s
        f1 = f(x0)
        x0[k] += 2*s
        f2 = f(x0)
        x0[k] -= s
        
        # finite difference value
        grad_FD = (f2-f1)/2s

        # @printf "%.12e - %.12e - %.12e\n" s grad_FD abs(grad_FD-grad_exact[k])/abs(grad_exact[k])
        @test abs(grad_exact[k] - grad_FD)/abs(grad_FD) < 2e-4
    end
end