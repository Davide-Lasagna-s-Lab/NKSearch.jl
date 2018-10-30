# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
import GMRES

export search!

# Arguments
# ---------
# G    : nonlinear propagator  - obeys `G(x, (0, T))` where `x` is modified in place
# L    : linearised propagator - obeys `L(Flows.couple(x, y), (0, T))` where `x` 
#        and `y` are modified in place
# S    : spatial shift operator - obeys `S(x, s)` where `x` is shifted by `s`
# dG   : derivative of `G(x, (0, T))` wrt to `T` - obeys `dG(out, x)`, where 
#        `out` gets overwritten
# dS   : derivative of `S` wrt to `s` - obeys `dS(out, x)` where `out` gets
#        overwritten
# z0   : initial guess vector, gets overwritten
# opts : search options (see src/options.jl)
function search!(G, L, S, dG, dS,
                   z0::MVector{X, N},
                 opts::Options=Options()) where {X, N}
    # display nice header
    opts.verbose && display_header(opts.io)

    # allocate memory
    b   = similar(z0)                   # right hand side
    tmp = similar(z0[1])                # temporary
    A   = MMatrix(G, L, S, dG, dS, z0)  # Newton update equation matrix operator

    # calculate initial error
    e_norm = e_norm_λ(G, S, z0, z0, 0.0, tmp)

    # display status if verbose
    opts.verbose && display_status(opts.io,    # input/output channel
                                   0,          # iteration number
                                   0,          # total norm of correction
                                   0,          # period correction
                                   0,          # shift correction
                                   sum(z0.T),  # current period
                                   z0.s,       # current shift
                                   e_norm,     # error norm after step
                                   0.0,        # step length
                                   0.0)        # GMRES residual norm

    # newton iterations loop
    for iter = 1:opts.maxiter

        # update Newton update matrix operator and right hand side
        update!(A, b, z0, opts)

        # solve system by overwriting b in place
        b, res_err_norm = GMRES.gmres!(A, b,
                                       opts.gmres_rtol,
                                       opts.gmres_maxiter,
                                       opts.gmres_verbose)

        # perform line search
        λ, e_norm = linesearch(G, S, z0, b, opts, tmp)

        # actually apply correction
        z0 .+= λ.*b

        # correction norm
        δx_norm = sum(norm(b[i])^2 for i = 1:N)

        # display status if verbose
        opts.verbose && display_status(opts.io,      # input/output channel
                                       iter,         # iteration number
                                       δx_norm,      # total norm of correction
                                       sum(b.T),     # period correction
                                       b.s,          # shift correction
                                       sum(z0.T),    # new period
                                       z0.s,         # new shift
                                       e_norm,       # error norm after step
                                       λ,            # step length
                                       res_err_norm) # GMRES residual error

        # tolerances reached
        e_norm  < opts.e_tol && break # norm of error
        δx_norm < opts.x_tol && break # norm of orbit correction
    end

    # return input
    return nothing
end