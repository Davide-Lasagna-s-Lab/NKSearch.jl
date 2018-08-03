using Base.Test
using Flows, KS, asis

@testset "utils                                  " begin
    @test asis._next_k(1, 3) == 2
    @test asis._next_k(2, 3) == 3
    @test asis._next_k(3, 3) == 1
    @test asis._prev_k(1, 3) == 3
    @test asis._prev_k(2, 3) == 1
    @test asis._prev_k(3, 3) == 2
    
    @test asis._next_k(1, 1) == 1
    @test asis._prev_k(1, 1) == 1
end

@testset "test f and g! for single shooting      " begin
    # parameters
    ν        = (2π/30)^2
    Δt       = 1e-2*ν
    T        = 10*ν
    ISODD    = false
    n        = 20

    # seed rng
    srand(0)

    # system right hand side
    F = ForwardEquation(n, ν, ISODD)

    # integrator
    ϕ = flow(splitexim(F)..., 
                   CB3R2R3e(FTField(n, ISODD), :NL), TimeStepConstant(Δt))

    # define the stage cache
    cache = (RAMStageCache(4, FTField(n, ISODD)), )

    # land on attractor and construct tuple of initial conditions
    U₀ = (ϕ(FTField(n, ISODD, k->exp(2π*im*rand())/k), (0, 100*ν)), )

    # rhs
    L⁺ = LinearisedEquation(n, ν, ISODD, KS.AdjointMode())

    # integrators
    ψ⁺ = flow(splitexim(L⁺)..., 
                   CB3R2R3e(FTField(n, ISODD), :ADJ), TimeStepFromCache())

    # build cache for gradient evaluation
    f, g! = GradientCache(ϕ, F, KS.ddx!, ψ⁺, U₀, cache)

    # initial condition
    x0 = tovector(U₀, T, 0.1)


    # //// TEST GRADIENT WITH RESPECT TO COMPONENTS ////
    for k = 1:5
        f0 = f(x0)
        x0[k] += 1e-7
        f1 = f(x0)
        x0[k] -= 1e-7
        
        # finite difference value
        grad_FD = (f1-f0)/1e-7

        # exact value
        grad_exact = similar(x0)
        g!(grad_exact, x0)

        @test abs(grad_exact[k] - grad_FD)/abs(grad_FD) < 4e-5
    end


    # //// TEST GRADIENT WITH RESPECT TO TIME ////
    f0 = f(x0)
    x0[end-1] += 1e-7
    f1 = f(x0)
    x0[end-1] -= 1e-7
    
    # finite difference value
    grad_FD = (f1-f0)/1e-7

    # exact value
    grad_exact = similar(x0)
    g!(grad_exact, x0)

    @test abs(grad_exact[end-1] - grad_FD)/abs(grad_FD) < 3e-6

    # //// TEST GRADIENT WITH RESPECT TO SHIFT ////
    f0 = f(x0)
    x0[end] += 1e-7
    f1 = f(x0)
    x0[end] -= 1e-7
    
    # finite difference value
    grad_FD = (f1-f0)/1e-7

    # exact value
    grad_exact = similar(x0)
    g!(grad_exact, x0)

    @test abs(grad_exact[end] - grad_FD)/abs(grad_FD) < 3e-6
end

# for some reason julia crashes if this is a test set
# @testset "test f and g! for multiple shooting    " begin
    # parameters
    ν        = (2π/30)^2
    Δt       = 1e-2*ν
    T        = 10*ν
    ISODD    = false
    n        = 20
    N        =  5

    # seed rng
    srand(0)

    # system right hand side
    F = ForwardEquation(n, ν, ISODD)

    # integrator
    ϕ = flow(splitexim(F)..., 
             CB3R2R3e(FTField(n, ISODD), :NL), TimeStepConstant(Δt))

    # define the stage cache
    cache = ntuple(i->RAMStageCache(4, FTField(n, ISODD)), N)

    # land on attractor
    U₀ = ϕ(FTField(n, ISODD, k->exp(2π*im*rand())/k), (0, 100*ν))

    # construct tuple of initial conditions, noting that we do not want 
    # to have a continuous trajectory, but rather several chunks.
    V = ntuple(i->ϕ(copy(U₀), (0, i*T/N + rand()*ν)), N);

    # rhs
    L⁺ = LinearisedEquation(n, ν, ISODD, KS.AdjointMode())

    # integrators
    ψ⁺ = flow(splitexim(L⁺)..., 
              CB3R2R3e(FTField(n, ISODD), :ADJ), TimeStepFromCache())

    # build cache for gradient evaluation
    f, g! = GradientCache(ϕ, F, KS.ddx!, ψ⁺, V, cache);
    
    # initial condition
    x0 = tovector(V, T, 0.1)

    # //// TEST GRADIENT WITH RESPECT TO THE FIRST FIVE COMPONENTS OF EVERY SHOOTING STAGE ////
    for skip in [0, 40, 80, 120, 160]
        for k = 1:5
            f0 = f(x0)
            x0[k + skip] += 1e-7
            f1 = f(x0)
            x0[k + skip] -= 1e-7
            
            # finite difference value
            grad_FD = (f1-f0)/1e-7

            # exact value
            grad_exact = similar(x0)
            g!(grad_exact, x0)

            @test abs(grad_exact[k + skip] - grad_FD)/abs(grad_FD) < 4e-5
        end
    end

    # //// TEST GRADIENT WITH RESPECT TO TIME ////
    f0 = f(x0)
    x0[end-1] += 1e-7
    f1 = f(x0)
    x0[end-1] -= 1e-7
    
    # finite difference value
    grad_FD = (f1-f0)/1e-7

    # exact value
    grad_exact = similar(x0)
    g!(grad_exact, x0)

    @test abs(grad_exact[end-1] - grad_FD)/abs(grad_FD) < 1e-5

    # //// TEST GRADIENT WITH RESPECT TO SHIFT ////
    f0 = f(x0)
    x0[end] += 1e-7
    f1 = f(x0)
    x0[end] -= 1e-7
    
    # finite difference value
    grad_FD = (f1-f0)/1e-7

    # exact value
    grad_exact = similar(x0)
    g!(grad_exact, x0)

    @test abs(grad_exact[end] - grad_FD)/abs(grad_FD) < 4e-6
# end