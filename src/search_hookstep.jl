# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

# line search method implementation
function _search_hs!(G, L, S, D, z::MVector{X, N, NS},  opts) where {X, N, NS}
    # display nice header
    opts.verbose && display_header_hs(opts.io, z)

    # allocate memory
    b   = similar(z)              # right hand side
    tmp = similar(z[1])           # temporary
    A   = MMatrix(G, L, S, D, z)  # Newton update equation matrix operator

    # calculate initial error
    e_norm = e_norm_λ(G, S, z, z, 0.0, tmp)

    # init
    tr_radius = opts.tr_radius_init

    # display status if verbose
    opts.verbose && display_status(opts.io,   # input/output channel
                                   0,         # iteration number
                                   0,         # total norm of correction
                                   e_norm,    # error norm after step
                                   1,         # actual/predicted reduction ratio
                                   tr_radius, # trust region radius after step
                                   0)         # GMRES residual error

    # newton iterations loop
    for iter = 1:opts.maxiter

        # update Newton update matrix operator and right hand side
        update!(A, b, opts)

        # solve system by overwriting b in place
        b, res_err_norm = GMRES.gmres!(A, b,
                                       tr_radius,
                                       opts.gmres_rtol,
                                       opts.gmres_maxiter,
                                       opts.gmres_verbose)

        # calc actual reductions
        e_norm_curr = e_norm_λ(G, S, z, b, 0.0, tmp)
        e_norm_next = e_norm_λ(G, S, z, b, 1.0, tmp)
        actual = _a - _b

        # calc predicted reduction
        predicted =

        # calc ratio
        rho = actual/predicted

        # trust region update
        if rho < 1/4
            tr_radius *= 1/4
        elseif rho > 3/4 && hits_boundary
            tr_radius = min(2*tr_radius, opts.tr_radius_max)
        end

        # solution update if reduction is large enough
        if rho > opts.eta
            z .= z .+ b
            e_norm = e_norm_next
        else
            e_norm = e_norm_curr
        end

        # calc correction norm
        dz_norm = norm(b)

        # display status if verbose
        opts.verbose && display_status(opts.io,      # input/output channel
                                       iter,         # iteration number
                                       dz_norm,      # total norm of correction
                                       e_norm,       # error norm after step
                                       rho,          # actual/predicted reduction ratio
                                       tr_radius,    # trust region radius after step
                                       res_err_norm) # GMRES residual error

        # tolerances reached
         e_norm <  opts.e_norm_tol && break
        dx_norm < opts.dx_norm_tol && break
    end

    # return input
    return z
end