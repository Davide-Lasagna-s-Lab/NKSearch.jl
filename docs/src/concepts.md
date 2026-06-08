```@meta
DocTestSetup = quote
    using NKSearch
end
```

# Concepts

This page explains the ideas you need to set up a search. None of it requires
reading the source.

## Multiple shooting

A naive formulation represents a periodic orbit by a single point ``x_0`` and
asks for ``G(x_0, T) = x_0``. For long or strongly unstable orbits this is
badly conditioned: a small change in ``x_0`` can produce a huge change in
``G(x_0, T)``.

Multiple shooting splits the orbit into ``N`` **segments** and stores one
**seed** state per segment, ``x_1, \dots, x_N``. Each seed is integrated only
over a fraction ``T/N`` of the period, and the unknowns are tied together by
matching conditions: the end of segment ``i`` must equal the start of segment
``i+1`` (with the last segment wrapping back to the first, up to a shift for a
relative periodic orbit). Shorter segments mean shorter integrations and a
better-conditioned linear system.

## The `MVector` unknown

All of the unknowns are packed into a single [`MVector`](@ref): the ``N`` seeds
plus the scalars (period ``T``, and optionally a shift ``s``).

```julia
z = MVector((x1, x2, x3), T)        # periodic orbit:          3 seeds + period
z = MVector((x1, x2, x3), T, s)     # relative periodic orbit: 3 seeds + period + shift
```

The seeds may be plain `Vector`s or any state type that supports `dot`,
`similar`, `zero`, and broadcasting (for example the field types in `Flows`).
`MVector` behaves as an `AbstractVector{Float64}`, and [`tovector`](@ref) /
[`fromvector!`](@ref) convert to and from a flat `Vector{Float64}` when needed.

```jldoctest
julia> z = MVector(([1.0, 2.0], [3.0, 4.0]), 6.28);

julia> nsegments(z), z.d
(2, (6.28,))

julia> tovector(z)
5-element Vector{Float64}:
 1.0
 2.0
 3.0
 4.0
 6.28
```

If your seeds come from a stored trajectory of `M` samples, use
[`find_number_of_segments`](@ref) to pick an `N` that divides `M` evenly while
keeping each segment within a target duration.

## Phase conditions

A periodic orbit has no preferred starting time: shifting the whole orbit along
itself gives another valid solution. A relative periodic orbit additionally has
no preferred spatial phase. These degeneracies would make the Newton system
singular, so they are removed by **phase-locking constraints**:

- the *time* phase is fixed using the right-hand side ``f`` of the ODE — the
  `F` argument to [`search!`](@ref), with the convention `F(out, x)`;
- the *spatial* phase (relative periodic orbits only) is fixed using the shift
  generator — the `dS` argument, with the convention `dS(out, x)`.

You do not implement the constraints themselves; you only provide `F` (and `dS`),
and NKSearch builds the constraints from them.

## The operators you provide

[`search!`](@ref) is matrix-free: instead of a Jacobian, you pass operators that
*act* on states.

| Operator | Role | Call signature |
|----------|------|----------------|
| `G` | nonlinear flow | `G(x, (0, T))` advances `x` in place |
| `L` | linearised flow | `L(couple(x, y), (0, T))` advances base `x` and perturbation `y` |
| `F` | ODE right-hand side | `F(out, x)` writes the time derivative |
| `S` | spatial shift (RPO only) | `S(x, s)` shifts `x` by `s` |
| `dS` | shift generator (RPO only) | `dS(out, x)` writes the spatial phase direction |

`G` and `L` typically come from `Flows` (`flow(...)` and `flow(couple(...))`).
If you do not have a hand-coded linearisation `L`, wrap `G` in a [`JFOp`](@ref),
which approximates ``L`` with a finite difference of ``G``:

```julia
L = JFOp(G, similar_state)   # matrix-free, finite-difference linearisation
```

## Indexing and layout conventions

- Seeds are ordered along the orbit: `z[i]` is the `i`-th seed, segment `i`
  spans time `((i-1)·T/N, i·T/N)`, and segment `N` connects back to segment `1`.
- The scalar unknowns are `z.d`: `z.d[1]` is the period `T`, and `z.d[2]` is the
  shift `s` when present (`nsegments` counts seeds, not scalars).
- In the flat layout from [`tovector`](@ref), the `N` seeds come first
  (concatenated) and the scalars last.

## Common pitfalls

- **Wrong `F` signature.** `search!` expects `F(out, x)`. If your right-hand
  side is `f(t, x, dxdt)` (the `Flows` convention), wrap it:
  `(out, x) -> f(0, x, out)`.
- **Threads vs. method.** The `_iterative` methods parallelise the segments
  across tasks and are intended to run with one thread per segment; the
  `_direct` methods must run single-threaded (they will error otherwise). See
  [Solver methods](@ref).
- **Guess too far from a real orbit.** Newton converges locally. Start from a
  reasonable guess (e.g. a recurrence detected in a long trajectory) and prefer
  a trust-region method far from the solution.
- **Period sign.** A line/trust step can momentarily drive `T` negative; the
  solvers guard against the resulting invalid time spans, but a guess with a
  realistic period helps avoid wasted iterations.
