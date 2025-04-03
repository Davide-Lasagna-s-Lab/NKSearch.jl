# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import Base.Threads: nthreads
using Printf

# trust region method implementation
function _search_hookstep!(Gs, Ls, S, D, z, cache, opts)
    # display nice header
    opts.verbose && display_header_hks(opts.io, z)

    # allocate memory
    b    = similar(z)                           # right hand side
    dz   = similar(z)                           # temporary
    tmps = ntuple(i->similar(z[1]), nthreads()) # one temporary for each thread

    # calculate initial error
    e_norm = e_norm_λ(Gs, S, z, z, 0.0, tmps)

    # init
    tr_radius = opts.tr_radius_init

    # display status if verbose
    opts.verbose && display_status_hks(opts.io,
                                      0,
                                      :start,
                                      0,
                                      e_norm,
                                      0,
                                      tr_radius, 0.0, 0)

    status = :maxiter_reached

    # avoid doing work if tolerance is already satisfied
    if e_norm <  opts.e_norm_tol
        status = :converged
        return status
    end

    # newton iterations loop
    for iter = 1:opts.maxiter
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # UPDATE CACHE
        update!(cache, b, z, opts)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # SOLVE TRUST REGION PROBLEM
        hits_boundary, which, step, gmres_res, gmres_it = solve_tr_subproblem!(dz, b, z, cache, tr_radius, opts)

        # calc actual reductions
        e_norm_curr = e_norm_λ(Gs, S, z, dz, 0.0, tmps)
        e_norm_next = e_norm_λ(Gs, S, z, dz, 1.0, tmps)
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
            display_status_hks(opts.io,
                               iter,
                               which,
                               dz_norm,
                               e_norm,
                               rho,
                               tr_radius, gmres_res, gmres_it)
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
function solve_tr_subproblem!(dz::MVector, b::MVector, z::MVector, cache, tr_radius::Real, opts::Options)

    if opts.method == :tr_direct
        return solve_dogleg_subproblem!(dz::MVector, b::MVector, z::MVector, cache, tr_radius::Real, opts::Options)
    end

    if opts.method == :tr_iterative
        return solve_hookstep_subproblem!(dz::MVector, b::MVector, z::MVector, cache, tr_radius::Real, opts::Options)
    end

    # this should not happen as we restrict the method in the Options struct
    throw(ArgumentError("panic!"))
end

function solve_hookstep_subproblem!(dz::MVector, b::MVector, z::MVector, cache, tr_radius::Real, opts::Options)
    # solve optimisation problem (this is always using GMRES)
    dz .*= 0.0 # TODO: better initial guess?
    dz, res_err_norm, n_iter = _solve(dz, cache, b, tr_radius, opts)

    # we consider a newton step only if we are within the trust region
    # and we have managed to reduce the error at least as much as required
    # by the gmres stopping tolerance. If the trust region is small, it might
    # not be possible to make res_err_norm smaller than required, regardless
    # of the number of iterations. This is a symptom of the fact that
    # the GMRES solution is affected by the trust region size and we can thus 
    # use this info to decide whether we want to increase or decrease it.
    if norm(dz) < tr_radius * (1 + 1e-6) && res_err_norm <= opts.gmres_rtol
        return false, :newton, norm(dz), res_err_norm, n_iter
    else 
        return true,  :hkstep, tr_radius, res_err_norm, n_iter
    end
end

function solve_dogleg_subproblem!(dz::MVector, b::MVector, z::MVector, cache, tr_radius::Real, opts::Options)
    # ~~~ GET NEWTON STEP ~~~
    dz .*= 0.0
    dz, res_err_norm = _solve(dz, cache, b, opts)

    if norm(dz_N) < tr_radius
        return false, :newton, 1.0, 0.0, 0
    end
    dz_N = copy(dz)

    # ~~~ GET CAUCHY STEP ~~~
    grad = cache.A'*tovector(b)
    den  = cache.A*grad

    dz_C = fromvector!(copy(b), grad)
    dz_C .*= (norm(grad)/norm(den))^2

    if norm(dz_C) > tr_radius
        dz .= dz_C
        fact = tr_radius / norm(dz_C)
        dz .*= fact
        return true, :cauchy, fact, 0.0, 0
    end

    # DOGLEG in case things are bad!
    dz_N_minus_dz_C   = dz_N # alias
    dz_N_minus_dz_C .-= dz_C
    tau = _solve_tr_boundary!(dz_C, dz_N_minus_dz_C, tr_radius)
    dz .= dz_C .+ tau .* dz_N_minus_dz_C
    return true, :dogleg, tau, 0.0, 0
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