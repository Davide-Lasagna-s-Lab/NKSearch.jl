# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

using Parameters

export Options

# ~~~ SEARCH OPTIONS FOR NEWTON ITERATIONS ~~~

@with_kw struct Options
    # generic parameters
    method::Symbol       = :linesearch_iter  # search method
    maxiter::Int         = 10           # maximum newton iteration number
    io                   = stdout       # where to print stuff
    skipiter::Int        = 0            # skip iteration between displays
    verbose::Bool        = true         # print iteration status
    dz_norm_tol::Float64 = 1e-10        # tolerance on correction
    e_norm_tol::Float64  = 1e-10        # tolerance on residual
    ϵ::Float64           = 1e-6         # dt for finite difference approximation
                                        # of the derivative of the flow operator
    # line search parameters
    ls_maxiter::Int      = 10           # maximum number of line search iterations
    ls_rho::Float64      = 0.5          # line search step reduction factor

    # GMRES parameters
    gmres_maxiter::Int   = 10           # maximum number of GMRES iterations
    gmres_verbose::Bool  = true         # print GMRES iteration status
    gmres_rtol::Float64  = 1e-3         # GMRES relative stopping tolerance

    # trust_region algorithm parameters
    tr_radius_init::Float64 = 1        # initial trust region radius
    tr_radius_max::Float64 = 10^8      # maximum trust region radius
    eta::Float64 = 0.25                # maximum trust region radius
end