# Solver methods

[`search!`](@ref) supports four methods, selected with `Options(method = …)`.
They differ in two independent choices: how the Newton step is **globalized**
(line search vs. trust region) and how the linear system is **solved** (direct
LU factorisation vs. matrix-free GMRES).

| `method` | Globalization | Linear solve | Threads |
|----------|---------------|--------------|---------|
| `:ls_direct` (default) | line search | LU factorisation | single |
| `:ls_iterative` | line search | GMRES (matrix-free) | one per segment |
| `:tr_direct` | trust region (dogleg) | LU factorisation | single |
| `:tr_iterative` | trust region (hookstep) | GMRES (matrix-free) | one per segment |

## Choosing a linear solve

- **Direct (`_direct`).** The Jacobian is assembled and LU-factorised. This is
  simplest and exact, but forming the matrix costs `O(n)` flow linearisations
  per segment (where `n` is the state size), so it is only practical for small
  states. Direct methods run **single-threaded** and will raise an error if
  Julia is started with more than one thread.

- **Iterative (`_iterative`).** The Jacobian is never formed; GMRES uses only
  matrix–vector products, each one a linearised flow. This scales to large
  states (discretised PDEs) and parallelises the shooting segments across tasks.
  Run Julia with as many threads as segments. Tune the solve with
  `gmres_maxiter` and `gmres_rtol`.

## Choosing a globalization

- **Line search (`ls_`).** Takes the Newton direction and backtracks the step
  length until the residual decreases (`ls_maxiter`, `ls_rho`). Cheap per
  iteration; can struggle far from a solution.

- **Trust region (`tr_`).** Restricts the step to a region where the linear
  model is trusted, expanding or shrinking the radius based on how well the
  model predicted the actual reduction. More robust far from the solution.
  `:tr_direct` uses a dogleg step; `:tr_iterative` uses a hookstep
  (trust-region-constrained GMRES). Relevant options: `tr_radius_init`,
  `tr_radius_max`, `min_step`, `NR_lim`, `α`, `eta`.

## Rule of thumb

- Small system, want simplicity: `:ls_direct` or `:tr_direct`.
- Large system (matrix-free): `:ls_iterative` or `:tr_iterative`.
- Poor initial guess / robustness needed: prefer a `tr_` method.

## Convergence and return value

A search stops when the residual norm drops below `e_norm_tol`, when the step
norm drops below `dz_norm_tol`, when the trust-region step falls below
`min_step`, when a `callback` returns `true`, or when `maxiter` is reached. The
trust-region and hookstep methods return a status `Symbol` (`:converged`,
`:maxiter_reached`, `:min_step_reached`, `:callback_satisfied`); the line-search
method returns `nothing`. In all cases the orbit `z` is refined in place.

See [`Options`](@ref) for the full list of tunable parameters.
