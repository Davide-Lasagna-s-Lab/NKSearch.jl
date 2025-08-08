# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

using Parameters

import GMRES: GMRESTrace

export Options, GMRESTrace

# ~~~ SEARCH OPTIONS FOR NEWTON ITERATIONS ~~~
@with_kw struct Options{GT<:Union{Nothing, GMRESTrace}, W, CB}
    # generic parameters
    method::Symbol          = :ls_direct           # search method
    maxiter::Int            = 10                   # maximum newton iteration number
    io                      = stdout               # where to print stuff
    skipiter::Int           = 1                    # skip iteration between displays
    verbose::Bool           = true                 # print iteration status
    dz_norm_tol::Float64    = 1e-10                # tolerance on correction
    e_norm_tol::Float64     = 1e-10                # tolerance on residual
    fd_order::Int           = 2                    # use forward or central difference scheme
                                                   # to approximate the derivative of the flow
                                                   # operator
    ϵ::Float64              = 1e-6                 # dt for finite difference approximation
                                                   # of the derivative of the flow operator
    callback::CB            = (iter, z)->false     # function called at the end of each
                                                   # iteration, if it returns true then the
                                                   # search terminates

    # line search parameters
    ls_maxiter::Int         = 10                   # maximum number of line search iterations
    ls_rho::Float64         = 0.5                  # line search step reduction factor

    # GMRES parameters
    gmres_maxiter::Int      = 10                   # maximum number of GMRES iterations
    gmres_verbose::Bool     = true                 # print GMRES iteration status
    gmres_rtol::Float64     = 1e-3                 # GMRES relative stopping tolerance
    gmres_trace::GT         = nothing              # keep track of GMRES state between newton iterations
    gmres_start::W          = dz->(dz .*= 0.0; dz) # GMRES warm start based on previous Newton step

    # trust_region algorithm parameters
    min_step::Float64       = 1e-4
    α::Float64              = 1
    NR_lim::Float64         = 1e-8
    tr_radius_init::Float64 = 1                    # initial trust region radius
    tr_radius_max::Float64  = 10^8                 # maximum trust region radius
    eta::Float64            = 0.00                 # maximum trust region radius

    @assert method in (:tr_direct, :ls_direct, :ls_iterative, :tr_iterative)
    @assert skipiter > 0
    @assert fd_order in (1, 2)
end
