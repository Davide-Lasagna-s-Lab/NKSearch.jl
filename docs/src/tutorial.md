# Tutorial

This tutorial converges a known limit cycle from a deliberately wrong guess. It
uses the Viswanath (2001) system, whose limit cycle is the unit circle with
period ``2\pi``:

```math
\dot x = -y + \mu x (1 - r), \qquad
\dot y =  x + \mu y (1 - r), \qquad
r = \sqrt{x^2 + y^2}.
```

The code blocks below run during the documentation build, so they are known to
work with the current API. They require the `Flows` and `GMRES` packages (see
[Installation](@ref)).

## 1. Define the dynamics

`search!` needs the right-hand side `F` and, for the linearised flow, the
Jacobian of the right-hand side. We follow the `Flows` calling conventions:
`F(t, x, dxdt)` for the system and `Fjac(t, x, dxdt, v, dvdt)` for its
linearisation.

```@example tutorial
using NKSearch, Flows, LinearAlgebra

struct System
    μ::Float64
end

function (s::System)(t, u, dudt)
    x, y, μ = u[1], u[2], s.μ
    r = sqrt(x^2 + y^2)
    dudt[1] = -y + μ*x*(1 - r)
    dudt[2] =  x + μ*y*(1 - r)
    return dudt
end

struct SystemLinear
    μ::Float64
    J::Matrix{Float64}
end
SystemLinear(μ) = SystemLinear(μ, zeros(2, 2))

function (s::SystemLinear)(t, u, dudt, v, dvdt)
    x, y, μ = u[1], u[2], s.μ
    r = sqrt(x^2 + y^2)
    s.J[1,1] = μ*(1 - r - x^2/r); s.J[1,2] = -1 - μ*x*y/r
    s.J[2,1] =  1 - μ*x*y/r;      s.J[2,2] = μ*(1 - r - y^2/r)
    return mul!(dvdt, s.J, v)
end

μ = 1.0
F    = System(μ)
Fjac = SystemLinear(μ)
nothing # hide
```

## 2. Build the flow operators

`G` integrates the nonlinear system; `L` integrates the base state and a
perturbation together (`couple`). Here we use a fixed-step RK4 integrator.

```@example tutorial
G = flow(F,
         RK4(zeros(2), Flows.NormalMode()),
         TimeStepConstant(1e-3))

L = flow(couple(F, Fjac),
         RK4(couple(zeros(2), zeros(2)), Flows.NormalMode()),
         TimeStepConstant(1e-3))
nothing # hide
```

If you did not have `Fjac`, you could instead use a matrix-free linearisation
`L = JFOp(G, zeros(2))`.

## 3. Make an initial guess

We use two shooting segments. The seeds are placed at radius 2 (the true orbit
has radius 1), and the period guess is ``2\pi``.

```@example tutorial
z = MVector(([2.0, 0.0], [-2.0, 0.0]), 2π)
nsegments(z), z.d
```

## 4. Search

`F` is wrapped to the `F(out, x)` convention that `search!` expects. We use the
hookstep trust-region method with a matrix-free GMRES solve.

```@example tutorial
status = search!(G, L, (dudt, u) -> F(0, u, dudt), z,
                 Options(method      = :tr_iterative,
                         maxiter     = 25,
                         e_norm_tol  = 1e-12,
                         gmres_maxiter = 5,
                         verbose       = false,
                         gmres_verbose = false))
status
```

## 5. Check the result

`z` has been refined in place. The seed radii should be 1 and the period
``2\pi``:

```@example tutorial
(radii = map(norm, z.x), period = z.d[1])
```

```@example tutorial
@assert maximum(abs, map(norm, z.x) .- 1) < 1e-8
@assert abs(z.d[1] - 2π) < 1e-8
println("converged to the unit limit cycle")
```

## Where to go next

- Try other [Solver methods](@ref) by changing `method` (for a 2×2 system the
  `:ls_direct` and `:tr_direct` methods are fine and run single-threaded).
- For a relative periodic orbit, build the guess with a shift,
  `MVector((x1, x2), T, s)`, and call the six-argument
  `search!(G, L, S, F, dS, z, opts)`.
- Save and reload converged orbits with [`save_seeds`](@ref) /
  [`load_seeds!`](@ref).
