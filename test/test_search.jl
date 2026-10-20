@testset "search_linesearch                      " begin
    # define systems
    μ = 1.0
    F = System(μ)
    D = SystemLinear(μ)

    # define propagators
    G = flow(F,
             RK4(zeros(2), Flows.NormalMode()),
             TimeStepConstant(1e-3))
    L = flow(couple(F, D),
             RK4(couple(zeros(2), zeros(2)), Flows.NormalMode()),
             TimeStepConstant(1e-3))

    # define methods
    methods = (#:ls_direct,      # only works with single thread (julia -t 1)
               :ls_iterative,   
               #:tr_direct,      # only works with single thread (julia -t 1)
               :tr_iterative,
               )
    orderings = (:regular, :ashtari)

    for method in methods, ordering in orderings
        # define initial guess, a slightly perturbed orbit
        z = MVector(([2, 0.0], [-2, 0.0]), 2π)

        # search
        search!(G,
                L,
                (dxdt, x)->F(0, x, dxdt),
                z,
                Options(maxiter=25,
                        dz_norm_tol=1e-8,
                        gmres_verbose=false,
                        e_norm_tol=1e-10,   # (below the accuracy tested, 1e-9)
                        gmres_maxiter=5,
                        verbose=true,
                        tr_radius_init=0.001,
                        method=method,
                        ϵ=1e-7,
                        gmres_start=dz->dz,
                        row_order=ordering))

        # solution is a loop of unit radius and with T = 2π
        @test maximum( map(el->norm(el)-1, z.x) ) < 1e-9
        @test abs(z.d[1] - 2π ) < 1e-9
    end
end

@testset "search_tr_iterative_rejected_steps     " begin
    # define systems
    μ = 1.0
    F = System(μ)
    D = SystemLinear(μ)

    # define propagators
    G = flow(F,
             RK4(zeros(2), Flows.NormalMode()),
             TimeStepConstant(1e-3))
    L = flow(couple(F, D),
             RK4(couple(zeros(2), zeros(2)), Flows.NormalMode()),
             TimeStepConstant(1e-3))

    # a guess inside the loop with a large initial radius: the first steps are rejected, and the hookstep for the
    # smaller radius comes from the Krylov space of the last GMRES, with no GMRES iteration
    z  = MVector(([0.3, 0.0], [-0.3, 0.0]), 2π + 1)
    io = IOBuffer()
    status = search!(G,
                     L,
                     (dxdt, x)->F(0, x, dxdt),
                     z,
                     Options(maxiter=40,
                             dz_norm_tol=1e-12,
                             e_norm_tol=1e-10,
                             gmres_maxiter=5,
                             gmres_rtol=1e-6,
                             gmres_verbose=false,
                             verbose=true,
                             io=io,
                             tr_radius_init=10.0,
                             method=:tr_iterative,
                             ϵ=1e-7))

    # the rows of the table: rho in the fifth column, the GMRES iterations in the last
    rows     = filter(l -> occursin(r"^\| +\d+ +\| (newton|hkstep)", l), split(String(take!(io)), '\n'))
    rho      = [parse(Float64, split(l, '|')[6]) for l in rows]
    gmres_it = [parse(Int, split(l, '|')[9]) for l in rows]
    rejected = findall(rho[1:end-1] .<= 0)

    @test status == :converged
    @test maximum( map(el->norm(el)-1, z.x) ) < 1e-9
    @test abs(z.d[1] - 2π ) < 1e-9
    @test !isempty(rejected)
    @test all(gmres_it[rejected .+ 1] .== 0)
end

@testset "search_lbfgs                           " begin
    # define systems
    μ = 1.0
    F = System(μ)
    D = SystemLinear(μ)
    D_adj = SystemLinearAdjoint(μ)

    # define propagators (stage-cache style for StageIterCache)
    G = flow(F,
             RK4(zeros(2), Flows.NormalMode()),
             TimeStepConstant(1e-3))
    L = flow(TangentSystem(D),
             RK4(zeros(2), Flows.DiscreteMode(false)),
             TimeStepFromCache())

    # Adjoint flow for L-BFGS methods
    adj_flow = flow(
        AdjointTangentSystem(D_adj),
        RK4(zeros(2), Flows.DiscreteMode(true)),
        TimeStepFromCache())

    for method in (:lbfgs_opt,)
        @testset "$method" begin
            # define initial guess, a slightly perturbed orbit
            z = MVector(([2, 0.0], [-2, 0.0]), 2π)

            # search
            search!(G,
                    L,
                    adj_flow,
                    (dxdt, x)->F(0, x, dxdt),
                    z,
                    Options(maxiter=100,
                            dz_norm_tol=1e-8,
                            gmres_verbose=false,
                            e_norm_tol=1e-8,
                            gmres_maxiter=5,
                            verbose=true,
                            tr_radius_init=0.0001,
                            method=method,
                            ls_maxiter=20,
                            gmres_start=dz->dz,
                            lbfgs_memory=5));

            # solution is a loop of unit radius and with T = 2π
            @test maximum( map(el->norm(el)-1, z.x) ) < 1e-9
            @test abs(z.d[1] - 2π ) < 1e-9

        end
    end
end
