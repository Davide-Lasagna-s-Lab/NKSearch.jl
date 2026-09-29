# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export search!

# Each segment owns a deep-copied flow and its work arrays. Tasks are indexed
# by segment, not threadid(); the thread count need not equal the segment count.
# Shared derivative/shift callables must not mutate shared scratch storage.

# Arguments
# ---------
# G    : nonlinear propagator  - obeys `G(x, (0, T))` where `x` is modified in place
# L    : linearised propagator - obeys `L(Flows.couple(x, y), (0, T))` where `x`
#        and `y` are modified in place
# S    : spatial shift operator - obeys `S(x, s)` where `x` is shifted by `s`
# F    : the right hand side of the governing equations. Obeys `F(out, x)`, where
#        `out` gets overwritten
# dS   : derivative of `S` wrt to `s` - obeys `dS(out, x)` where `out` gets
#        overwritten
# z0   : initial guess vector, gets overwritten
# opts : search options (see src/options.jl)

"""
    search!(G, L, S, F, dS, z0::MVector{X,N,2}, opts=Options()) -> status
    search!(G, L,       F,     z0::MVector{X,N,1}, opts=Options()) -> status
    search!(G, L, L_adj, F, z0::MVector{X,N,1}, opts) -> status
    search!(G, L, L_adj, S, F, dS, z0::MVector{X,N,2}, opts) -> status

Refine the candidate orbit `z0` in place with a Newton–Krylov / multiple-
shooting iteration until convergence or `opts.maxiter` is reached.

Use the relative-orbit form to search for a **relative periodic orbit** (an orbit
closing up to a spatial shift, `z0` has a shift unknown, `NS == 2`), and the
ordinary-orbit form for an ordinary **periodic orbit** (`NS == 1`).

`z0` is overwritten. Line-search Newton methods return `nothing`; trust-region
and L-BFGS methods return a status symbol. Newton trust-region methods can
return `:converged` on a small correction even if closure is not below tolerance.
Always verify the final residual. Only the hookstep method honours a stopping
callback; L-BFGS callbacks are observers.

The overloads with `L_adj` require `opts.method == :lbfgs_opt`. They minimise
the squared shooting residual using Flows stage caches, a discrete tangent
`L(v, stages)` and a discrete adjoint `L_adj(w, stages)`. They do not accept
`JFOp` as a replacement for the adjoint.

# Arguments
- `G`: nonlinear flow operator. `G(x, (0, T))` advances state `x` in place
  over time span `(0, T)`.
- `L`: linearised flow operator. `L(Flows.couple(x, y), (0, T))` advances the
  base state `x` and the perturbation `y` in place. To avoid writing a
  hand-coded linearisation, pass a [`JFOp`](@ref) built from `G`.
- `S`: spatial shift operator (relative periodic orbits only). `S(x, s)`
  shifts state `x` by `s` in place.
- `F`: right-hand side of the governing ODE. `F(out, x)` overwrites `out`
  with the time derivative at `x`; it sets the phase-locking constraint that
  removes the time-translation degeneracy.
- `dS`: generator of the spatial shift (relative periodic orbits only).
  `dS(out, x)` overwrites `out`, fixing the spatial phase.
- `z0::MVector`: initial guess; overwritten with the result. See
  [`MVector`](@ref).
- `opts::Options`: solver settings; see [`Options`](@ref).

Newton also calls `G(x, span, monitor)` to save a restart state for period
differences. The saved state must be restart-complete. Dynamics are assumed
autonomous because each segment starts at local time zero.

`G` and `L` (and `L_adj` when present) are deep-copied per segment, so the same operator
instance can be passed for all segments.

!!! note "Threading"
    The iterative methods (`:ls_iterative`, `:tr_iterative`) parallelise the
    shooting segments across tasks with separate caches. One or more threads
    may be used. The direct methods (`:ls_direct`, `:tr_direct`) require a
    single thread. Shared user callables must be safe for concurrent calls.

# Example
```julia
using NKSearch, Flows, LinearAlgebra

F = ...   # ODE right-hand side, callable as F(t, x, dxdt)
G = flow(F, RK4(zeros(2)), TimeStepConstant(1e-3))                      # nonlinear flow
L = flow(couple(F, Fjac), RK4(couple(zeros(2), zeros(2))), TimeStepConstant(1e-3))

z = MVector(([2.0, 0.0], [-2.0, 0.0]), 2π)   # 2-segment guess, period 2π
search!(G, L, (dxdt, x) -> F(0, x, dxdt), z,
        Options(method=:tr_iterative, maxiter=25))
```
See the manual for a complete, runnable tutorial.
"""
search!(G, L, S, F, dS, z0::MVector{X, N, 2}, opts::Options=Options()) where {X, N} =
    _search!(ntuple(i->deepcopy(G), nsegments(z0)),
             ntuple(i->deepcopy(L), nsegments(z0)), S, (F, dS), z0, opts)

# when we do not have shifts
search!(G, L, F, z0::MVector{X, N, 1}, opts::Options=Options()) where {X, N} =
    _search!(ntuple(i->deepcopy(G), nsegments(z0)), 
             ntuple(i->deepcopy(L), nsegments(z0)), nothing, (F, ), z0, opts)

# with adjoint flows (L-BFGS), with shift
search!(G, L, L_adj, S, F, dS, z0::MVector{X, N, 2}, opts::Options=Options()) where {X, N} =
    _search!(ntuple(i->deepcopy(G), nsegments(z0)),
             ntuple(i->deepcopy(L), nsegments(z0)),
             ntuple(i->deepcopy(L_adj), nsegments(z0)), S, (F, dS), z0, opts)

# with adjoint flows (L-BFGS), without shift
search!(G, L, L_adj, F, z0::MVector{X, N, 1}, opts::Options=Options()) where {X, N} =
    _search!(ntuple(i->deepcopy(G), nsegments(z0)),
             ntuple(i->deepcopy(L), nsegments(z0)),
             ntuple(i->deepcopy(L_adj), nsegments(z0)), nothing, (F, ), z0, opts)

# dispatch to correct method (without adjoint)
function _search!(Gs, Ls, S, D, z0::MVector{X, N, NS}, opts) where {X, N, NS}
    return (  opts.method == :ls_direct
            ? _search_linesearch!(Gs, Ls, S, D, z0, DirectSolCache(Gs, Ls, S, D, z0, opts), opts)
            : opts.method == :ls_iterative
            ? _search_linesearch!(Gs, Ls, S, D, z0, IterSolCache(Gs, Ls, S, D, z0, opts), opts)
            : opts.method == :tr_direct
            ? _search_trustregion!(Gs, Ls, S, D, z0, DirectSolCache(Gs, Ls, S, D, z0, opts), opts)
            : opts.method == :tr_iterative
            ? _search_hookstep!(Gs, Ls, S, D, z0, IterSolCache(Gs, Ls, S, D, z0, opts), opts)
            : throw(ArgumentError("unknown method: $(opts.method)")))
end

# dispatch with adjoint flows (L-BFGS methods)
function _search!(Gs, Ls, Ls_adj, S, D, z0::MVector{X, N, NS}, opts) where {X, N, NS}
    opts.method == :lbfgs_opt || throw(ArgumentError("unknown method: $(opts.method)"))
    fwd_cache = StageIterCache(Gs, Ls, S, D, z0)
    adj_cache = AdjointIterSolCache(Ls_adj, D, S, fwd_cache.xT, fwd_cache.dxTdT, fwd_cache.z0, fwd_cache.tmp, fwd_cache.stage_caches)
    return _search_lbfgs_opt!(Gs, Ls, S, D, z0, fwd_cache, adj_cache, opts)
end
