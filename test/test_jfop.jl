@testset "search_jfop                            " begin

    # define systems
    μ = 1.0
    F = System(μ)
    D = SystemLinear(μ)

    # define propagators
    G = flow(F,
             RK4(zeros(2), :NORMAL),
             TimeStepConstant(1e-3))

    # exact linearised propagator
    LEX = flow(couple(F, D),
             RK4(couple(zeros(2), zeros(2)), :NORMAL),
             TimeStepConstant(1e-3))

    # finite difference approximation
    LFD = JFOp(G, zeros(2))

    # initial conditions
    x1 = [1.0, 1.0]
    y1 = [1.0, 1.0]
    x2 = [1.0, 1.0]
    y2 = [1.0, 1.0]

    # we have not yet seen this
    @test NKSearch.has_seen(LFD, (0, 1), x2) == false
    @test LFD.tmps[1] != G(copy(x2), (0, 1))

    # calc using FD approx
    LFD(Flows.couple(copy(x1), y1), (0, 1))
    LEX(Flows.couple(copy(x2), y2), (0, 1))

    # they must be similar
    @test norm(y1 - y2)/norm(y1) < 1e-6

    # we do not want to repeat the calculation a second time
    @test NKSearch.has_seen(LFD, (0, 1), x2)         == true
    @test NKSearch.has_seen(LFD, (0, 2), x2)         == false
    @test NKSearch.has_seen(LFD, (0, 2), [1.0, 2.0]) == false
end