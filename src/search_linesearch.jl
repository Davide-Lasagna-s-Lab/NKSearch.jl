# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

import Flows
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

function e_norm_λ(G,
                  S,
                 z0::MVector{X, N, NS},
                 δz::MVector{X, N, NS},
                  λ::Real,
                tmp::X) where {X, N, NS}

    # initialize
    val_λ = zero(norm(tmp)^2)

    # sum all error contributions
    for i = 1:N
        # set initial condition
        tmp  .= z0[i] .+ λ.*δz[i]

        # and propagation time
        Ti = (z0.d[1] + λ*δz.d[1])/N

        # actual propagation
        G(tmp, (0, Ti))

        # last segment is shifted (if we have a shift)
        NS == 2 && i == N && S(tmp, z0.d[2] + λ*δz.d[2])

        # calc difference
        tmp .-= z0[i%N+1] .+ λ.*δz[i%N+1]

        # add to error
        val_λ += norm(tmp)^2
    end

    return val_λ
end

function linesearch(G, S, z0::MVector{X, N}, δz::MVector{X, N}, opts::Options, tmp::X) where {X, N}
    # current error
    val_0 = e_norm_λ(G, S, z0, δz, 0.0, tmp)

    # start with full Newton step
    λ = 1.0

    # initialize this variable
    val_λ = λ*val_0

    for iter = 1:opts.ls_maxiter
        # calculate error
        try
            val_λ = e_norm_λ(G, S, z0, δz, λ, tmp)
        catch err
            # We might end up in a situation where the
            # new time span has nehative length. In
            # such a case, we might just continue
            if !isa(err, Flows.InvalidSpanError)
                rethrow(err)
            end
        end

        # accept any reduction of error
        val_λ < val_0 && return λ, val_λ

        # ~ otherwise attempt with shorter step ~
        λ *= opts.ls_rho
    end

    error("maximum number of line search iterations reached")
end