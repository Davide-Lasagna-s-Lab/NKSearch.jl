# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
using Printf

# trust region method implementation
function _search_hookstep!(Gs, Ls, S, D, z, cache, opts)
    # display nice header
    opts.verbose && display_header_hks(opts.io, z)

    # allocate memory
    b    = similar(z)                             # right hand side
    dz   = similar(z); dz .*= 0.0                 # temporary
    tmps = ntuple(i->similar(z[1]), nsegments(z)) # one temporary for each segment

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

    # the Krylov space of the last GMRES (iterative method), kept after a rejected step: the orbit, the Jacobian and
    # the right-hand side have not changed, so the hookstep for the smaller radius needs no new GMRES
    K        = nothing
    rejected = false

    # newton iterations loop
    for iter = 1:opts.maxiter
        # callback
        opts.callback(iter, z, copy(b), e_norm, 0.0, 1.0, z.d[1], cache) && (status = :callback_satisfied; break)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # UPDATE CACHE (not after a rejected step)
        rejected || update!(cache, b, z, opts)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # SOLVE TRUST REGION PROBLEM (after a rejected step, in the Krylov space of the last GMRES)
        hits_boundary, which, step, gmres_res, gmres_it, K =
            solve_tr_subproblem!(opts.gmres_start(dz), b, z, cache, tr_radius, opts, rejected ? K : nothing)

        # calc actual reductions
        e_norm_curr = e_norm_λ(Gs, S, z, dz, 0.0, tmps)
        e_norm_next = e_norm_λ(Gs, S, z, dz, 1.0, tmps)
        actual = e_norm_curr^2 - e_norm_next^2

        # calc predicted reduction
        predicted = norm(cache * dz)^2

        # calc ratio
        rho = actual/predicted

        if e_norm_curr > opts.NR_lim
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
                rejected = false
            else
                e_norm = e_norm_curr
                rejected = K !== nothing
            end
        else
            z .= z .+ dz
            e_norm = e_norm_next
            rejected = false
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
                               tr_radius, gmres_res/norm(b), gmres_it)
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

# Solve the Trust Region optimisation subproblem; also return the Krylov space of the
# iterative method (nothing for the direct one), and take the one of the last GMRES, K,
# after a rejected step
function solve_tr_subproblem!(dz::MVector, b::MVector, z::MVector, cache, tr_radius::Real, opts::Options, K=nothing)

    if opts.method == :tr_direct
        return (solve_dogleg_subproblem!(dz::MVector, b::MVector, z::MVector, cache, tr_radius::Real, opts::Options)..., nothing)
    end

    if opts.method == :tr_iterative
        return solve_hookstep_subproblem!(dz::MVector, b::MVector, z::MVector, cache, tr_radius::Real, opts::Options, K)
    end

    # this should not happen as we restrict the method in the Options struct
    throw(ArgumentError("panic!"))
end

function solve_hookstep_subproblem!(dz::MVector, b::MVector, z::MVector, cache, tr_radius::Real, opts::Options, K=nothing)
    # solve optimisation problem (this is always using GMRES, which stops when the
    # unconstrained step in its Krylov space meets the tolerance, and returns the
    # hookstep in that space). After a rejected step, the hookstep for the new radius
    # in the Krylov space K of the last GMRES, with no product with the Jacobian
    if K === nothing
        dz, res_err_norm, n_iter, K = _solve(dz, cache, b, tr_radius, opts)
    else
        dz, res_err_norm = hookstep!(dz, K, tr_radius)
        n_iter = 0
    end

    # we consider a newton step only if we are within the trust region
    # and we have managed to reduce the error at least as much as required
    # by the gmres stopping tolerance (relative to the right-hand side). If the
    # trust region is small, it might not be possible to make res_err_norm smaller
    # than required, regardless of the number of iterations. This is a symptom of
    # the fact that the GMRES solution is affected by the trust region size and we
    # can thus use this info to decide whether we want to increase or decrease it.
    if norm(dz) < tr_radius * (1 + 1e-6) && res_err_norm <= opts.gmres_rtol * norm(b)
        return false, :newton, norm(dz), res_err_norm, n_iter, K
    else
        return true,  :hkstep, tr_radius, res_err_norm, n_iter, K
    end
end

function solve_dogleg_subproblem!(dz::MVector, b::MVector, z::MVector, cache, tr_radius::Real, opts::Options)
    # ~~~ GET NEWTON STEP ~~~
    dz, res_err_norm = _solve(dz, cache, b, opts)
    dz_N = copy(dz)

    # if the full Newton step is inside the trust region, take it
    if norm(dz_N) < tr_radius
        return false, :newton, 1.0, 0.0, 0
    end

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
