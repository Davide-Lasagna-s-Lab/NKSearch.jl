# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

using Printf

# line search method implementation
function _search_hookstep!(G, L, S, D, z, cache, opts)
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
        predicted = norm(fromvector!(b, cache.A*tovector(dz)))^2

        # calc ratio
        rho = actual/predicted

        if e_norm_curr > 1e-6
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
         e_norm <  opts.e_norm_tol && break
        dz_norm < opts.dz_norm_tol && break
    end

    # return input
    return z
end