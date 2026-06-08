# NKSearch.jl

A Newton–Krylov solver for finding **periodic orbits** and **relative periodic
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
(and ``s``) are all unknowns, solved together by a globalized Newton iteration.
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
- [Solver methods](@ref) — choosing among the four `method` options.
- [API reference](@ref) — the exported types and functions.
