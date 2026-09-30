# NKSearch.jl

Newton–Krylov and adjoint-based L-BFGS searches for finding **periodic orbits** and **relative periodic
orbits** of dynamical systems, using a multiple-shooting formulation.

Given a system `ẋ = f(x)` and an approximate guess for a closed orbit, NKSearch
refines the guess until it satisfies the periodicity condition to a chosen
tolerance, while simultaneously correcting the **period** `T` and, for a
relative periodic orbit, a **spatial shift** `s`.

## What it solves

A periodic orbit satisfies

```math
G(x_0, T) = x_0,
```

where ``G(\cdot, T)`` is the flow map that integrates the dynamics for a time
``T``. A *relative* periodic orbit closes only up to a continuous spatial
symmetry,

```math
S(G(x_0, T), s) = x_0,
```

with ``S`` the shift operator. The orbit point ``x_0`` and the scalars ``T``
(and ``s``) are all unknowns. Newton methods solve their shooting equations;
L-BFGS minimises the squared shooting residual using an adjoint gradient.
The Jacobian can be assembled and factorised directly, or applied matrix-free
and inverted with GMRES — the latter scales to the large states typical of
discretised PDEs.

## When should I use this?

- You have a time integrator and want to converge a periodic (or relative
  periodic) orbit from an approximate guess.
- The state may be large (e.g. a discretised PDE), so a matrix-free option is
  attractive.
- You want multiple shooting for robustness on long or sensitive orbits.

## Installation

NKSearch and its solver dependencies are not in the General registry; install
them from the [Davide-Lasagna-s-Lab](https://github.com/Davide-Lasagna-s-Lab)
organisation:

```julia
using Pkg
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/Flows.jl")
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/GMRES.jl")
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/NKSearch.jl")
```

## Where to next

- [Concepts](@ref) — multiple shooting, the [`MVector`](@ref) unknown, phase
  conditions, and the operator interface you provide.
- [Tutorial](@ref) — a complete, runnable example converging a limit cycle.
- [Solver methods](@ref) — choosing among Newton and L-BFGS methods.
- [API reference](@ref) — the exported types and functions.

## Adjoint-based L-BFGS search

Select `Options(method=:lbfgs_opt)` to minimise half the squared
multiple-shooting mismatch. This method updates the orbit points and period
using a limited-memory approximation to the inverse Hessian, without an inner
GMRES solve. It requires a stage-caching nonlinear Flows integrator and its
discrete tangent and adjoint flows:

```julia
# G, L and L_adj must use compatible integration stages.
status = search!(G, L, L_adj, F, z,
    Options(method=:lbfgs_opt, lbfgs_memory=10, maxiter=200,
            ls_maxiter=20, ls_rho=0.5, e_norm_tol=1e-8))
```

`F(out, x)` supplies the vector field. `JFOp` alone cannot supply the adjoint
needed here; use a Newton–Krylov method when only finite-difference
Jacobian–vector products are available. A small gradient can indicate a
nonzero-residual local minimum, so acceptance must be based on the orbit
closure residual. Stage storage can dominate memory for long trajectories.

See [the L-BFGS guide](lbfgs.md) for the objective, a complete example,
operator contracts and current implementation limitations.
