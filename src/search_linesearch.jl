# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

import LinearAlgebra: norm

# line search method implementation
function _search_linesearch!(G, L, S, D, z0, A, opts)
    # display nice header
    opts.verbose && display_header_ls(opts.io, z0)

    # allocate memory
    b   = similar(z0)                   # right hand side
    tmp = similar(z0[1])                # temporary

    # calculate initial error
    e_norm = e_norm_λ(G, S, z0, z0, 0.0, tmp)

    # display status if verbose
    opts.verbose && display_status_ls(opts.io,
                                      0,
                                      0,
                                      z0.d,
                                      e_norm,
                                      0.0,
                                      0.0)

    # newton iterations loop
    for iter = 1:opts.maxiter

        # update Newton update matrix operator and right hand side
        update!(A, b, z0, opts)

        # solve system by overwriting b in place
        b, res_err_norm = _solve(A, b, opts)

        # perform line search
        λ, e_norm = linesearch(G, S, z0, b, opts, tmp)

        # actually apply correction
        z0 .+= λ.*b

        # correction norm
        dz_norm = norm(b)

        # display status if verbose
        opts.verbose && display_status_ls(opts.io,
                                          iter,
                                          dz_norm,
                                          z0.d,
                                          e_norm,
                                          λ,
                                          res_err_norm)

        # tolerances reached
         e_norm  < opts.e_norm_tol && break # norm of error
        dz_norm < opts.dz_norm_tol && break # norm of orbit correction
    end

    # return input
    return nothing
end