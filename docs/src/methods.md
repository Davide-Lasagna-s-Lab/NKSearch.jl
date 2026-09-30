# Solver methods

[`search!`](@ref) supports four Newton methods and an L-BFGS method, selected with `Options(method = …)`.
The Newton methods differ in two independent choices: how the Newton step is **globalized**
(line search vs. trust region) and how the linear system is **solved** (direct
LU factorisation vs. matrix-free GMRES).

| `method` | Globalization | Linear solve | Threads |
|----------|---------------|--------------|---------|
| `:ls_direct` (default) | line search | LU factorisation | single |
| `:ls_iterative` | line search | GMRES (matrix-free) | segment tasks |
| `:tr_direct` | trust region (dogleg) | LU factorisation | single |
| `:tr_iterative` | trust region (hookstep) | GMRES (matrix-free) | segment tasks |
| `:lbfgs_opt` | objective backtracking | none; adjoint gradient | segment tasks |

## Choosing a linear solve

- **Direct (`_direct`).** The Jacobian is assembled and LU-factorised. This is
  a direct solve of the assembled numerical linearisation, but forming the matrix costs `O(n)` flow linearisations
  per segment (where `n` is the state size), so it is only practical for small
  states. Direct methods run **single-threaded** and will raise an error if
  Julia is started with more than one thread.

- **Iterative (`_iterative`).** The Jacobian is never formed; GMRES uses only
  matrix–vector products, each one a linearised flow. This scales to large
  states (discretised PDEs) and parallelises the shooting segments across tasks.
  Each segment owns its caches; the number of threads need not match the
  number of segments. Tune the solve with
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

| Method | Return value | Callback |
|---|---|---|
| `:ls_direct`, `:ls_iterative` | `nothing` | not called |
| `:tr_direct` | status symbol | not called |
| `:tr_iterative` | status symbol | eight arguments; `true` stops |
| `:lbfgs_opt` | status symbol | seven arguments; return value ignored |

Trust-region Newton drivers return `:converged` for either a small residual
or a small correction. Thus this status alone does not certify orbit closure.
They can also return `:maxiter_reached` or `:min_step_reached`; hookstep adds
`:callback_satisfied`. L-BFGS returns `:converged` only for its residual test,
and labels a small update `:min_step_reached`.

Always reintegrate the final seeds and check the matching residuals. The
`e_norm_type` option currently affects L-BFGS reporting/stopping only; the
Newton drivers use the Euclidean residual. See [`Options`](@ref) for the
actual callback signatures and the information available at each call.

## L-BFGS optimisation

`:lbfgs_opt` minimises the shooting mismatch with an adjoint gradient and a
limited-memory quasi-Newton direction. There is no inner linear solve;
`gmres_*` and `tr_*` options do not control this method. Use the adjoint
overload `search!(G, L, L_adj, F, z, opts)`.

Its callback has seven arguments and its return value is ignored. The method
returns `:converged`, `:min_step_reached` or `:maxiter_reached`; it does not
return `:callback_satisfied`. The objective uses the Euclidean `MVector` norm,
while `e_norm_type` selects the reported/stopping residual norm.

Read [L-BFGS search](lbfgs.md) before selecting it: the current backtracking
fallback does not guarantee descent when every trial is rejected.
