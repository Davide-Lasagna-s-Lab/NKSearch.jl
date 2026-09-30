# API reference

```@meta
CurrentModule = NKSearch
```

## The search

```@docs
search!
Options
```

## The unknown vector

```@docs
MVector
nsegments
tovector
fromvector!
find_number_of_segments
```

## Matrix-free linearisation

```@docs
JFOp
```

## Saving and loading orbits

```@docs
save_seeds
load_seeds!
```

## Sensitivity analysis (experimental)

```@docs
make_adjoint_problem
```

## L-BFGS search overloads

```julia
search!(G, L, L_adj, F, z, Options(method=:lbfgs_opt))
search!(G, L, L_adj, S, F, dS, z, Options(method=:lbfgs_opt))
```

These require stage-cached discrete tangent and adjoint flows, not `JFOp`.
See [L-BFGS search](lbfgs.md) for their contracts and callback signature.

### Stage-cache internals

These types are implementation tools; the public search driver constructs them.

```@docs
StageIterCache
AdjointIterSolCache
OptLBFGSCache
```
