# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export search!

function search!(G, L, S, dG, dS,
                 caches::NTuple{N, C},
                     z0::MVector{X, N},
                   opts::Options=Options()) where {X, N, C}
    # display nice header
    opts.verbose && display_header(opts.io)

    # allocate memory
    b   = similar(z0)                           # right hand side
    tmp = similar(z0[1])                        # temporary
    A   = MMatrix(G, L, S, dG, dS, caches, z0)  # Newton update equation matrix operator

    # newton iterations loop
    for iter = 1:opts.maxiter

        # update Newton update matrix operator and right hand side
        update!(A, b, z0)

        # solve system by overwriting b in place
        GMRES.gmres!(A, b, opts.gmres_rtol, opts.gmres_maxiter, opts.gmres_verbose)

        # perform line search
        λ, e_norm = linesearch(G, S, z0, b, opts, tmp)

        # actually apply correction
        z0 .+= λ.*b

        # correction norm
        δx_norm = sum(norm(b[i])^2 for i = 1:N)

        # display status if verbose
        opts.verbose && display_status(opts.io, # input/output channel
                                       iter,    # iteration number
                                       δx_norm, # total norm of correction
                                       δT,      # period correction
                                       δs,      # shift correction
                                       sum(T),  # new period
                                       s,       # new shift
                                       e_norm,  # error norm after step
                                       λ)       # step length

        # tolerances reached
        e_norm  < opts.e_tol && break # norm of error
        δx_norm < opts.x_tol && break # norm of orbit correction
    end

    # return input
    return nothing
end