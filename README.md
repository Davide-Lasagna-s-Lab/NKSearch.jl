# NKSearch.jl

A Newton–Krylov solver for finding **periodic orbits** and **relative periodic
orbits** of dynamical systems, using a multiple-shooting formulation.

Given a system `ẋ = f(x)` and a rough initial guess for a closed orbit,
NKSearch refines the guess until it satisfies the periodicity condition to a
chosen tolerance, simultaneously correcting the **period** `T` (and, for a
relative periodic orbit, a **spatial shift** `s`).

## What it solves

A periodic orbit satisfies `G(x₀, T) = x₀`, where `G(·, T)` is the flow map
that integrates the dynamics for a time `T`. A *relative* periodic orbit closes
only up to a continuous spatial symmetry: `S(G(x₀, T), s) = x₀`. Both the orbit
point `x₀` and the scalars `T` (and `s`) are unknowns.

NKSearch poses this as a nonlinear root-finding problem and solves it with a
globalized Newton iteration. The Jacobian can be assembled and factorised
directly, or applied matrix-free and inverted with GMRES — the latter scales to
the large states typical of discretised PDEs.

## When should I use this?

- You have a time integrator for a dynamical system and want to converge a
  periodic (or relative periodic) orbit from an approximate guess.
- The state may be large (e.g. a discretised PDE), so you want a matrix-free
  (Jacobian-free) option.
- You want multiple-shooting for robustness on long or sensitive orbits.

If you only need to integrate trajectories (not converge orbits), you just need
a time-stepper such as [`Flows`](https://github.com/Davide-Lasagna-s-Lab/Flows.jl);
NKSearch sits on top of one.

## Installation

NKSearch and its solver dependencies are not in the General registry; install
them directly from the
[Davide-Lasagna-s-Lab](https://github.com/Davide-Lasagna-s-Lab) organisation:

```julia
using Pkg
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/Flows.jl")
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/GMRES.jl")
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/NKSearch.jl")
```

## Key concepts

### Multiple shooting

Instead of representing the orbit by a single point, NKSearch splits it into
`N` **segments** and stores one **seed** state per segment. Periodicity becomes
a set of matching conditions between the end of one segment and the start of the
next. Shorter segments make the linearised problem better conditioned, which
helps on long or strongly unstable orbits. All of this is bundled into a single
[`MVector`](https://github.com/Davide-Lasagna-s-Lab/NKSearch.jl):

```julia
z = MVector((x₁, x₂, x₃), T)        # periodic orbit:          3 seeds + period
z = MVector((x₁, x₂, x₃), T, s)     # relative periodic orbit: 3 seeds + period + shift
```

### Phase conditions

A periodic orbit has no preferred starting time, and a relative periodic orbit
has no preferred spatial phase. These degeneracies are removed by **phase-locking
constraints** built from the right-hand side `f` (time phase) and the shift
generator (spatial phase) — the `F` and `dS` arguments to `search!`.

### Operators you provide

| Operator | Role | Call signature |
|----------|------|----------------|
| `G` | nonlinear flow | `G(x, (0, T))` advances `x` in place |
| `L` | linearised flow | `L(couple(x, y), (0, T))` advances base `x` and perturbation `y` |
| `F` | ODE right-hand side | `F(out, x)` writes the time derivative (time phase condition) |
| `S` | spatial shift (RPO only) | `S(x, s)` shifts `x` by `s` |
| `dS` | shift generator (RPO only) | `dS(out, x)` writes the spatial phase direction |

If you do not have a hand-coded linearisation `L`, wrap `G` in a
[`JFOp`](https://github.com/Davide-Lasagna-s-Lab/NKSearch.jl) for a matrix-free,
finite-difference approximation.

### Solver methods

Set `Options(method = …)`:

| `method` | Globalization | Linear solve | Threads |
|----------|---------------|--------------|---------|
| `:ls_direct` | line search | LU factorisation | single |
| `:ls_iterative` | line search | GMRES (matrix-free) | one per segment |
| `:tr_direct` | trust region (dogleg) | LU factorisation | single |
| `:tr_iterative` | trust region (hookstep) | GMRES (matrix-free) | one per segment |

Use a `_direct` method for small states where forming the Jacobian is cheap;
use an `_iterative` method for large states. Trust-region (`tr_`) methods tend
to be more robust far from the solution than line search.

## Minimal working example

The Viswanath (2001) system has the unit circle as a stable limit cycle of
period `2π`. Starting from a deliberately wrong two-segment guess (radius 2),
NKSearch converges it:

```julia
using NKSearch, Flows, LinearAlgebra

# ẋ = -y + μx(1 - r),  ẏ = x + μy(1 - r),  r = √(x² + y²)
struct System; μ::Float64; end
function (s::System)(t, u, dudt)
    x, y, μ = u[1], u[2], s.μ
    r = sqrt(x^2 + y^2)
    dudt[1] = -y + μ*x*(1 - r)
    dudt[2] =  x + μ*y*(1 - r)
    return dudt
end

# linearisation of the right-hand side
struct SystemLinear; μ::Float64; J::Matrix{Float64}; end
SystemLinear(μ) = SystemLinear(μ, zeros(2, 2))
function (s::SystemLinear)(t, u, dudt, v, dvdt)
    x, y, μ = u[1], u[2], s.μ
    r = sqrt(x^2 + y^2)
    s.J[1,1] = μ*(1 - r - x^2/r); s.J[1,2] = -1 - μ*x*y/r
    s.J[2,1] =  1 - μ*x*y/r;      s.J[2,2] = μ*(1 - r - y^2/r)
    return mul!(dvdt, s.J, v)
end

μ = 1.0
F = System(μ)
Fjac = SystemLinear(μ)

# nonlinear and linearised flow maps (RK4, fixed step)
G = flow(F, RK4(zeros(2), Flows.NormalMode()), TimeStepConstant(1e-3))
L = flow(couple(F, Fjac),
         RK4(couple(zeros(2), zeros(2)), Flows.NormalMode()),
         TimeStepConstant(1e-3))

# two-segment guess: points at radius 2, period 2π
z = MVector(([2.0, 0.0], [-2.0, 0.0]), 2π)

# converge it (F is wrapped to the F(out, x) convention search! expects)
search!(G, L, (dudt, u) -> F(0, u, dudt), z,
        Options(method=:tr_iterative, maxiter=25, e_norm_tol=1e-12,
                gmres_maxiter=5, verbose=false, gmres_verbose=false))

# z now holds a loop of unit radius with period 2π
@show maximum(map(norm, z.x))   # ≈ 1.0
@show z.d[1]                    # ≈ 2π
```

> **Threads.** The iterative methods parallelise the shooting segments. Run
> Julia with `JULIA_NUM_THREADS` equal to the number of segments (2 here) to use
> them, or with a single thread for the `_direct` methods.

## Documentation

Full documentation (concepts, tutorial, API reference) is built with
[Documenter.jl](https://documenter.juliadocs.org/) from the [`docs/`](docs/)
folder:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The generated site is written to `docs/build/`.

## Status

This is a research package. The public API is the set of exported symbols
documented above; internal types (the solution caches, the sensitivity
problems) may change. Sensitivity analysis (tangent/adjoint of a converged
orbit) is available but still being stabilised.
