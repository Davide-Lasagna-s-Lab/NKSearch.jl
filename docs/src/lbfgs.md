# L-BFGS search

## What is minimised?

Split an orbit of period ``T`` into ``N`` equal time intervals. The unknown
``z=(x_1,\ldots,x_N,T)`` contains their initial conditions and the period.
Writing ``\Phi_\tau`` for evolution over ``\tau=T/N``, define

```math
r_i(z)=x_{i+1}-\Phi_{T/N}(x_i),\qquad x_{N+1}=x_1,
\qquad \mathcal J(z)=\frac12\sum_{i=1}^N\|r_i(z)\|_2^2.
```

Each residual measures a mismatch between adjacent trajectory segments.
A zero objective joins the segments into a periodic trajectory. This is a
shooting objective, not a time integral of the differential-equation residual.
The norm is the one supplied by the state and `MVector` dot products; with
ordinary real vectors it is Euclidean. Scale state variables appropriately
before searching if their units or magnitudes differ strongly.

For a relative periodic orbit the last residual is
``r_N=x_1-S(\Phi_{T/N}(x_N),s)`` and the shift ``s`` is another unknown.
The adjoint implementation uses the inverse shift as the transpose, which
requires an orthogonal shift under the chosen dot product.

## Gradient and discrete adjoint

Let ``A_i=D\Phi_{T/N}(x_i)``. For an ordinary periodic orbit,

```math
\nabla_{x_i}\mathcal J=r_{i-1}-A_i^\mathsf{T}r_i,
\qquad
\partial_T\mathcal J=-\frac1N\sum_i f(\Phi_{T/N}(x_i))^\mathsf{T}r_i,
```

with cyclic indexing. One backward adjoint propagation per segment supplies
``A_i^\mathsf{T}r_i`` without forming ``A_i``. Forward trajectories save their
integration stages in `Flows.RAMStageCache`; the discrete adjoint reuses those
stages in reverse. A continuous adjoint integrated independently is not an
interchangeable implementation of this interface.

The period column currently uses the endpoint vector field ``f(\Phi)``.
That is the continuous flow identity; at finite timestep it need not be the
exact derivative of every possible numerical time-stepping policy. Check a
directional finite difference of the objective and refine the timestep when
accurate gradients are important.

The internal augmented operator also contains phase rows, but the residual's
scalar entries are zero. Thus the L-BFGS objective does not penalise an absolute
time or spatial phase: equivalent phase-shifted solutions remain possible.
Do not infer a unique orbit phase from convergence.

## How the step is constructed

The direction is ``p_k=-H_k\nabla\mathcal J_k``. L-BFGS represents ``H_k``
through at most `lbfgs_memory` recent pairs
``s_k=z_{k+1}-z_k`` and ``y_k=\nabla\mathcal J_{k+1}-\nabla\mathcal J_k``.
A two-loop recursion applies this representation. The implementation skips
pairs with nonpositive curvature ``s_k^\mathsf{T}y_k`` or negligible changes.
It does not assemble a Hessian and does not run GMRES.

History storage is proportional to `lbfgs_memory * length(z)`. In addition,
the forward stage caches scale with the number of integration steps, stages
and state variables; “limited memory” does not mean trajectories are free.

Backtracking starts at step length one and multiplies it by `ls_rho` after a
rejected trial. Acceptance requires strict objective decrease, not a Wolfe or
Armijo condition. The code compares ``\|r\|^2``; the omitted factor one-half
does not change that acceptance test.

## Operator interface

```julia
# Periodic orbit
search!(G, L, L_adj, F, z, Options(method=:lbfgs_opt))

# Relative periodic orbit: z = MVector(seeds, T, s)
search!(G, L, L_adj, S, F, dS, z, Options(method=:lbfgs_opt))
```

- `G` is a Flows nonlinear flow in `NormalMode`, supporting stage recording.
- `L(v, stages)` uses `DiscreteMode(false)` and `TimeStepFromCache()`.
- `L_adj(w, stages)` uses `DiscreteMode(true)` and `TimeStepFromCache()`.
- `F(out, x)` overwrites `out` with the nonlinear vector field.
- `S(x,s)` shifts in place; `dS(out,x)` supplies its infinitesimal generator.

All flows must use the same spatial discretisation and compatible integration
stages. The driver deep-copies flows per segment. A plain callable flow map
or `JFOp` is not a substitute for this stage-cache contract. In particular,
a forward finite-difference Jacobian action does not provide its transpose.

## Complete periodic-orbit example

This smooth two-dimensional system has the unit circle as a limit cycle with
period ``2\pi``. The tangent and adjoint right-hand sides below use the same
Jacobian, with a transpose for the adjoint. Run this in a fresh Julia session.

```@example lbfgs
using NKSearch, Flows, LinearAlgebra

function rhs!(t, x, out)
    a = 1 - dot(x, x)
    out[1] = -x[2] + a*x[1]
    out[2] =  x[1] + a*x[2]
    return out
end

struct CircleLinear
    adjoint::Bool
end
function (f::CircleLinear)(t, x, v, out)
    a = 1 - dot(x, x)
    j11 = a - 2*x[1]^2
    j22 = a - 2*x[2]^2
    j12 = -1 - 2*x[1]*x[2]
    j21 =  1 - 2*x[1]*x[2]
    b, c = f.adjoint ? (j21, j12) : (j12, j21)
    out[1] = j11*v[1] + b*v[2]
    out[2] = c*v[1] + j22*v[2]
    return out
end

G = flow(rhs!, RK4(zeros(2), Flows.NormalMode()),
         TimeStepConstant(1e-3))
L = flow(CircleLinear(false), RK4(zeros(2), Flows.DiscreteMode(false)),
         TimeStepFromCache())
L_adj = flow(CircleLinear(true), RK4(zeros(2), Flows.DiscreteMode(true)),
             TimeStepFromCache())
F = (out, x) -> rhs!(0.0, x, out)
z = MVector(([1.1, 0.0], [-1.1, 0.0]), 6.3)

history = NamedTuple[]
record = (iteration, z, residual, error, gradient, step, period) ->
    push!(history, (; iteration, error, gradient, step, period))
log = IOBuffer()  # Keep the iteration log independently of display capture.
opts = Options(method=:lbfgs_opt, lbfgs_memory=5, maxiter=200,
               ls_maxiter=20, ls_rho=0.5, e_norm_tol=1e-8,
               dz_norm_tol=1e-12, callback=record, io=log)
status = search!(G, L, L_adj, F, z, opts)
print(String(take!(log)))
@show status z.d[1] norm.(z.x)

# Verify closure independently of the optimiser's status.
errors = map(eachindex(z.x)) do i
    endpoint = copy(z.x[i])
    G(endpoint, (0.0, z.d[1]/length(z.x)))
    norm(endpoint - z.x[mod1(i+1, length(z.x))])
end
@show errors
```

The example was checked locally: it returned `:converged`, a period of
`6.2831853133`, and segment closure errors of approximately `5.3e-9`.
These are example results, not a guarantee for another dynamical system.

## Options and interpretation

| Option | Role in L-BFGS |
|---|---|
| `lbfgs_memory` | Number of stored correction pairs; use a positive integer |
| `maxiter` | Maximum optimisation iterations |
| `ls_maxiter`, `ls_rho` | Backtracking trials and contraction factor |
| `e_norm_tol`, `e_norm_type` | Closure stopping tolerance and norm |
| `dz_norm_tol` | Stop if the accepted update becomes too small |
| `callback` | Seven-argument progress observer shown above |
| `verbose`, `skipiter`, `io` | Progress output |

`gmres_*`, `tr_*`, `fd_order` and `ϵ` do not set the L-BFGS gradient calculation.
The objective always uses the full squared norm even if `e_norm_type` selects
a different residual norm for reporting and stopping.

`:converged` means the residual satisfies its tolerance. `:min_step_reached`
means stagnation, not a periodic orbit. `:maxiter_reached` means the iteration
budget was exhausted. A small gradient with a nonzero residual can be a
stationary point of the least-squares problem and is not successful closure.
Equilibria also satisfy the periodicity equation; closure alone does not
establish a nontrivial orbit or its least period.

## Current implementation limitations

- If all backtracking trials fail, the current code returns step length one
  and applies it. Consequently objective decrease is not guaranteed on that
  iteration. Inspect the recorded residual history.
- Callback return values are ignored; callbacks observe the search but do
  not terminate it. The callback is also called at iteration zero.
- Once the residual is below tolerance, gradient evaluation is skipped.
  The reported gradient on that iteration can therefore be stale; an
  already-converged initial guess can have an uninitialised gradient report.
- Rejected trials that raise an exception should not be assumed to leave the
  candidate untouched: trial evaluation mutates it before the flow call.
- Validate the relative-orbit shift generator sign against your definition of
  `S`: an adjoint identity alone tests a transpose pairing, not whether both
  operators differentiate the intended nonlinear residual.

These describe the present implementation, not promised convergence guarantees.
Before a large calculation, verify the tangent/adjoint dot-product identity,
compare the objective directional derivative with finite differences, and
check closure again at a smaller timestep.
