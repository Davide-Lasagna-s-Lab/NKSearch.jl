# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

using Parameters

export Options

# ~~~ SEARCH OPTIONS FOR NEWTON ITERATIONS ~~~
"""
    Options(; kwargs...)

Configure a Newton or L-BFGS shooting search. Only override the needed options.

# Method and stopping
- `method=:ls_direct`: `:ls_direct`, `:ls_iterative`, `:tr_direct`,
  `:tr_iterative`, or `:lbfgs_opt`. The first four are Newton methods;
  L-BFGS requires the `search!` overload with a discrete adjoint flow.
- `maxiter=10`: maximum outer iterations.
- `e_norm_tol=1e-10`: residual stopping tolerance.
- `dz_norm_tol=1e-10`: correction stopping tolerance. Newton trust-region
  drivers label a small correction `:converged`, even without small closure;
  L-BFGS returns `:min_step_reached`. Independently check the final residual.
- `e_norm_type=:euclidean`: L-BFGS reporting/stopping norm, either
  `:euclidean` or `:max_segment`. The objective remains the full squared norm.
  Newton drivers currently use the Euclidean residual regardless of this option.
- `verbose=true`, `io=stdout`, `skipiter=1`: progress output destination/cadence.

# Newton period differences
- `ϵ=1e-6`: time increment for the finite-difference period column.
- `fd_order=2`: forward (`1`) or centred (`2`) period difference.
  These do not control `JFOp.epsilon` or the L-BFGS adjoint gradient.
- `row_order=:ashtari`: row arrangement of the iterative Newton operator;
  `:regular` retains segment order. This is not a physical change of unknowns.

# Line search
- `ls_maxiter=10`, `ls_rho=0.5`: maximum trials and contraction factor.
- `ls_method=:backtracking`: the supported line-search choice.
  Both Newton line search and L-BFGS seek strict decrease, not Wolfe conditions.
  L-BFGS currently falls back to a full step if all trials fail.

# GMRES (iterative Newton only)
- `gmres_maxiter=10`, `gmres_rtol=1e-3`, `gmres_verbose=true`: inner solve.
- `gmres_callback=nothing`: passed to GMRES.
- `gmres_start=dz -> (dz .*= 0.0; dz)`: initialises the hookstep correction.

# Trust region (Newton only)
- `tr_radius_init=1.0`, `tr_radius_max=1e8`: initial and maximum radius.
- `min_step=1e-4`: trust-region subproblem step threshold.
- `NR_lim=1e-8`: residual threshold below which a full Newton step is used.
- `eta=0.0`: acceptance ratio threshold.
- `α=1.0`: relaxation of the near-root step in the direct trust-region driver.

# L-BFGS
- `lbfgs_memory=10`: positive number of stored correction pairs.
  The forward integration stages require additional memory.

# Callbacks
The default `(args...) -> false` accepts the method-specific arguments:
- Hookstep: `callback(iter, z, rhs, error, 0.0, 1.0, T, cache)` runs before
  the iteration's cache update; `true` stops with `:callback_satisfied`.
  The copied `rhs` is stale and is uninitialised on the first call.
- L-BFGS: `callback(iter, z, residual, error, gradient_norm, step, T)` runs
  at iteration zero and after updates. Its return value is ignored. The
  gradient can be stale when the residual already meets tolerance.
- Direct trust region and Newton line search do not invoke `callback`.

# Example
```julia
opts = Options(method=:tr_iterative, maxiter=25,
               e_norm_tol=1e-10, gmres_maxiter=20)
```
"""
@with_kw struct Options{GT, W, CB}
    # generic parameters
    method::Symbol          = :ls_direct           # search method
    maxiter::Int            = 10                   # maximum newton iteration number
    io                      = stdout               # where to print stuff
    skipiter::Int           = 1                    # skip iteration between displays
    verbose::Bool           = true                 # print iteration status
    dz_norm_tol::Float64    = 1e-10                # tolerance on correction
    e_norm_tol::Float64     = 1e-10                # tolerance on residual
    e_norm_type::Symbol     = :euclidean           # L-BFGS stopping norm: :euclidean or :max_segment
    fd_order::Int           = 2                    # use forward or central difference scheme
                                                   # to approximate the derivative of the flow
                                                   # operator
    ϵ::Float64              = 1e-6                 # dt for finite difference approximation
                                                   # of the derivative of the flow operator
    callback::CB            = (args...)->false     # user-provided callback function
    row_order::Symbol       = :ashtari             # row ordering for Newton system

    # line search parameters
    ls_method::Symbol       = :backtracking        # line search method
    ls_maxiter::Int         = 10                   # maximum number of line search iterations
    ls_rho::Float64         = 0.5                  # line search step reduction factor

    # GMRES parameters
    gmres_maxiter::Int      = 10                   # maximum number of GMRES iterations
    gmres_verbose::Bool     = true                 # print GMRES iteration status
    gmres_rtol::Float64     = 1e-3                 # GMRES relative stopping tolerance
    gmres_callback::GT      = (args...)->false     # GMRES callback function
    gmres_start::W          = dz->(dz .*= 0.0; dz) # GMRES warm start based on previous Newton step

    # trust_region algorithm parameters
    min_step::Float64       = 1e-4                 # minimum step tolerance
    α::Float64              = 1                    # over-relaxation factor for trust region update
    NR_lim::Float64         = 1e-8                 # maximum limit for newton region update
    tr_radius_init::Float64 = 1                    # initial trust region radius
    tr_radius_max::Float64  = 10^8                 # maximum trust region radius
    eta::Float64            = 0.00                 # maximum trust region radius

    # L-BFGS history; independent of GMRES and trust-region settings
    lbfgs_memory::Int       = 10                   # number of history vectors for L-BFGS

    @assert method in (:tr_direct, :ls_direct, :ls_iterative, :tr_iterative, :lbfgs_opt)
    @assert skipiter > 0
    @assert fd_order in (1, 2)
    @assert ls_method in (:backtracking,)
    @assert row_order in (:regular, :ashtari)
end
