# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

using Parameters

export Options

# ~~~ SEARCH OPTIONS FOR NEWTON ITERATIONS ~~~

@with_kw struct Options
    maxiter::Int        = 10           # maximum newton iteration number
    io                  = STDOUT       # where to print stuff
    skipiter::Int       = 0            # skip iteration between displays
    verbose::Bool       = true         # print iteration status
    x_tol::Float64      = 1e-10        # tolerance on initial state correction
    e_tol::Float64      = 1e-10        # tolerance on initial state correction
    ls_maxiter::Int     = 10           # maximum number of line search iterations 
    ls_rho::Float64     = 0.5          # line search step reduction factor
    gmres_maxiter::Int  = 10           # maximum number of GMRES iterations
    gmres_verbose::Bool = true         # print GMRES iteration status
    gmres_rtol::Float64 = 1e-3         # GMRES relative stopping tolerance
    ϵ::Float64          = 1e-6         # dt for finite difference approximation
                                       # of the derivative of the flow operator
    method::Symbol      = :linesearch  # search method 
end