function solve_tr_subproblem!(dz::MVector, z::MVector, cache, tr_radius::Real, opts::Options)

    # ~~~ GET NEWTON STEP ~~~
    dz_N, res_err_norm = _solve(cache, copy(dz), opts)

    if norm(dz_N) < tr_radius
        dz .= dz_N
        return false, :newton, 1.0
    end

    # ~~~ GET CAUCHY STEP ~~~
    grad = cache.A'*tovector(dz)
    den  = cache.A*grad

    dz_C = fromvector!(copy(dz), grad)
    dz_C .*= (norm(grad)/norm(den))^2

    if norm(dz_C) > tr_radius
        dz .= dz_C
        fact = tr_radius / norm(dz_C)
        dz .*= fact
        return true, :cauchy, fact
    end

    # DOGLEG in case things are bad!
    dz_N_minus_dz_C   = dz_N # alias
    dz_N_minus_dz_C .-= dz_C
    tau = _solve_tr_boundary!(dz_C, dz_N_minus_dz_C, tr_radius)
    dz .= dz_C .+ tau .* dz_N_minus_dz_C
    return true, :dogleg, tau
end

# Solve for the largest τ such that ||q + τ*p||^2 = tr_radius^2
function _solve_tr_boundary!(q, p, tr_radius::Real)
    # compute coefficients of the quadratic equation
    a = norm(p)^2
    b = 2*dot(q, p)
    c = norm(q)^2 - tr_radius^2
    # compute discriminant and then return positive (largest) root
    sq_discr = sqrt(b^2 - 4*a*c)
    return max(- b + sq_discr, - b - sq_discr)/2a
end