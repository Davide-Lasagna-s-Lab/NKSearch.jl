# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

import LinearAlgebra: norm
# solution for iterative method
_solve(A::MMatrix, b::MVector, opts::Options) =
    GMRES.gmres!(A, b, opts.gmres_rtol,
                       opts.gmres_maxiter,
                       opts.gmres_verbose)


# line search method implementation
function _search_linesearch!(G, L, S, D, z0, A, opts)
    # display nice header
    opts.verbose && display_header(opts.io, z0)

    # allocate memory
    b   = similar(z0)                   # right hand side
    tmp = similar(z0[1])                # temporary

    # calculate initial error
    e_norm = e_norm_λ(G, S, z0, z0, 0.0, tmp)

    # display status if verbose
    opts.verbose && display_status(opts.io, # input/output channel
                                   0,       # iteration number
                                   0,       # total norm of correction
                                   z0.d,    # current shifts
                                   e_norm,  # error norm after step
                                   0.0,     # step length
                                   0.0)     # GMRES residual norm

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
        opts.verbose && display_status(opts.io,      # input/output channel
                                       iter,         # iteration number
                                       dz_norm,      # total norm of correction
                                       z0.d,         # new period
                                       e_norm,       # error norm after step
                                       λ,            # step length
                                       res_err_norm) # GMRES residual error

        # tolerances reached
         e_norm  < opts.e_norm_tol && break # norm of error
        dz_norm < opts.dz_norm_tol && break # norm of orbit correction
    end

    # return input
    return nothing
end