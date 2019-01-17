# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

using Printf

# trust region method implementation
function _search_trustregion!(G, L, S, D, z, cache, opts)
    # display nice header
    opts.verbose && display_header_tr(opts.io, z)

    # allocate memory
    b   = similar(z)              # right hand side
    dz  = similar(z)              # temporary
    tmp = similar(z[1])           # temporary

    # calculate initial error
    e_norm = e_norm_λ(G, S, z, z, 0.0, tmp)

    # init
    tr_radius = opts.tr_radius_init

    # display status if verbose
    opts.verbose && display_status_tr(opts.io,
                                      0,
                                      :start,
                                      0,
                                      e_norm,
                                      0,
                                      tr_radius,
                                      0)

    status = :maxiter_reached

    # newton iterations loop
    for iter = 1:opts.maxiter
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # UPDATE CACHE
        update!(cache, dz, z, opts)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # SOLVE TRUST REGION PROBLEM
        hits_boundary, which, step = solve_tr_subproblem!(dz, z, cache, tr_radius, opts)

        # calc actual reductions
        e_norm_curr = e_norm_λ(G, S, z, dz, 0.0, tmp)
        e_norm_next = e_norm_λ(G, S, z, dz, 1.0, tmp)
        actual = e_norm_curr - e_norm_next

        # calc predicted reduction
        predicted = norm(cache * dz)^2

        # calc ratio
        rho = actual/predicted

        if e_norm_curr > 1e-7
            # trust region update
            if rho < 1/4
                tr_radius *= 1/4
            elseif rho > 3/4 && hits_boundary
                tr_radius = min(2*tr_radius, opts.tr_radius_max)
            end

            # solution update if reduction is large enough
            if rho > opts.eta
                z .= z .+ dz
                e_norm = e_norm_next
            else
                e_norm = e_norm_curr
            end
        else
            z .= z .+ dz
            e_norm = e_norm_next
        end

        dz_norm = norm(dz)

        # display status if verbose
        if opts.verbose && iter % opts.skipiter == 0
            display_status_tr(opts.io,
                              iter,
                              which,
                              step,
                              e_norm,
                              rho,
                              tr_radius,
                              dz_norm)
        end

        # tolerances reached
        if e_norm <  opts.e_norm_tol
            status = :converged
            break
        end
        if dz_norm < opts.dz_norm_tol
            status = :converged
            break
        end
        if step < opts.min_step
            status = :min_step_reached
            break
        end
    end

    # return input
    return status
end

# Solve the Trust Region optimisation subproblem
function solve_tr_subproblem!(dz::MVector, z::MVector, cache, tr_radius::Real, opts::Options)

    if opts.method == :tr_direct
        return solve_dogleg_subproblem!(dz::MVector, z::MVector, cache, tr_radius::Real, opts::Options)
    end

    if opts.method == :tr_iterative
        return solve_hookstep_subproblem!(dz::MVector, z::MVector, cache, tr_radius::Real, opts::Options)
    end

    # this should not happen as we restrict the method in the Options struct
    throw(ArgumentError("panic!"))
end

function solve_hookstep_subproblem!(dz::MVector, z::MVector, cache, tr_radius::Real, opts::Options)
    
    # solve optimisation problem (this is always using GMRES)
    dz, res_err_norm = _solve(cache, dz, tr_radius, opts)

    if norm(dz) < tr_radius
        return false, :newton, 1.0
    else 
        return true,  :hkstep, tr_radius
    end
end

function solve_dogleg_subproblem!(dz::MVector, z::MVector, cache, tr_radius::Real, opts::Options)
    
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