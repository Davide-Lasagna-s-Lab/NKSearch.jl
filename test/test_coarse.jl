@testset "coarse preconditioner                   " begin
    μ = 1.0
    F = System(μ)
    D = SystemLinear(μ)
    G = flow(F, RK4(zeros(2), Flows.NormalMode()), TimeStepConstant(1e-3))
    L = flow(couple(F, D), RK4(couple(zeros(2), zeros(2)), Flows.NormalMode()), TimeStepConstant(1e-3))
    Fp = (dxdt, x)->F(0, x, dxdt)

    # With k equal to the state dimension, W is the whole space, M⁻¹ = A⁻¹
    # and GMRES converges in one iteration.
    z = MVector(([1.2, 0.1], [-0.9, 0.3], [0.1, -1.1]), 6.0)
    its = Ref(0)
    opts = Options(method=:ls_iterative, coarse_dim=2, gmres_verbose=false, gmres_maxiter=10,
                   gmres_rtol=1e-12, gmres_callback=(it, args...)->(its[] = it; false))
    A = NKSearch.IterSolCache(ntuple(i->deepcopy(G), 3), ntuple(i->deepcopy(L), 3), nothing, (Fp,), z, opts)
    b = similar(z)
    NKSearch.update!(A, b, z, opts)
    x, res = NKSearch._solve(zero(z), A, b, opts)
    @test its[] == 1
    t = A*x; t .-= b
    @test norm(t) < 1e-10*norm(b)

    # M is the inverse of M⁻¹
    r = MVector((randn(2), randn(2), randn(2)), randn())
    t = NKSearch.mul_precond!(similar(r), A.pc, NKSearch.ldiv_precond!(similar(r), A.pc, r)); t .-= r
    @test norm(t) < 1e-12*norm(r)

    # the search converges with a partial coarse space
    for k in (1, 2)
        z = MVector(([2, 0.0], [-2, 0.0]), 2π)
        search!(G, L, Fp, z, Options(method=:ls_iterative, coarse_dim=k, maxiter=25,
                                     e_norm_tol=1e-8, dz_norm_tol=1e-8, gmres_maxiter=5,
                                     gmres_verbose=false, verbose=false, ϵ=1e-7, gmres_start=dz->dz))
        @test maximum(map(el->norm(el)-1, z.x)) < 1e-9
        @test abs(z.d[1] - 2π) < 1e-9
    end

    # The hookstep bounds the step in the original unknowns. With k equal to the
    # state dimension the Krylov space is the line of the Newton step, which the
    # hookstep shortens to the radius.
    z  = MVector(([1.2, 0.1], [-0.9, 0.3], [0.1, -1.1]), 6.0)
    xN = copy(x)
    Δ  = 1e-2*norm(xN)
    opts = Options(method=:tr_iterative, coarse_dim=2, gmres_verbose=false, gmres_maxiter=10, gmres_rtol=1e-12)
    A = NKSearch.IterSolCache(ntuple(i->deepcopy(G), 3), ntuple(i->deepcopy(L), 3), nothing, (Fp,), z, opts)
    NKSearch.update!(A, b, z, opts)
    x, res, _, K = NKSearch._solve(zero(z), A, b, Δ, opts)
    t = copy(x); t .-= (Δ/norm(xN)).*xN
    @test norm(t) < 1e-10*Δ

    # With a partial coarse space the step meets the radius when it binds, also
    # for another radius in the same Krylov space, and the residual returned is
    # the true one.
    opts = Options(method=:tr_iterative, coarse_dim=1, gmres_verbose=false, gmres_maxiter=20, gmres_rtol=1e-12)
    A = NKSearch.IterSolCache(ntuple(i->deepcopy(G), 3), ntuple(i->deepcopy(L), 3), nothing, (Fp,), z, opts)
    NKSearch.update!(A, b, z, opts)
    x, res, _, K = NKSearch._solve(zero(z), A, b, Δ, opts)
    for δ in (Δ, Δ/10)
        x, res = NKSearch._hookstep!(x, A.pc, K, δ)
        t = A*x; t .-= b
        @test abs(norm(x) - δ) < 1e-10*δ
        @test abs(norm(t) - res) < 1e-8*norm(b)
    end

    # the hookstep search converges with a partial coarse space
    for k in (1, 2)
        z = MVector(([2, 0.0], [-2, 0.0]), 2π + 0.5)
        status = search!(G, L, Fp, z, Options(method=:tr_iterative, coarse_dim=k, maxiter=40,
                                              e_norm_tol=1e-10, dz_norm_tol=1e-12, gmres_maxiter=5,
                                              gmres_rtol=1e-6, tr_radius_init=0.1,
                                              gmres_verbose=false, verbose=false, ϵ=1e-7))
        @test status == :converged
        @test maximum(map(el->norm(el)-1, z.x)) < 1e-9
        @test abs(z.d[1] - 2π) < 1e-9
    end
end
