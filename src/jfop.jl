# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export JFOp

"""
    JFOp(G, x, epsilon::Float64=1e-6) -> op

A matrix-free ("Jacobian-free") linearised flow operator, suitable as the
`L` argument of [`search!`](@ref) when no hand-coded linearisation is
available.

It approximates the action of the linearised flow on a perturbation `y`
with a finite-difference quotient of the nonlinear flow `G`:

```math
L\\{x\\} y \\approx \\frac{G(x + \\epsilon\\, y) - G(x)}{\\epsilon}
```

# Arguments
- `G`: nonlinear flow operator, callable as `G(x, span)` (in place).
- `x`: a sample state, used only to allocate internal temporaries
  (`similar(x)`).
- `epsilon::Float64`: finite-difference step.

# Usage
Call it like the exact linearised operator,
`op(Flows.couple(x, y), (0, T))`, which advances base state `x` and
perturbation `y` in place. The base trajectory `G(x, span)` is cached
(keyed by `span` and `x`) so it is not recomputed for repeated perturbations
about the same point.

!!! warning
    The quotient is a first-order forward difference regardless of
    `Options.fd_order`, and `x` must be real-valued.
"""
mutable struct JFOp{X, GT}
          G::GT                # operator
       tmps::NTuple{2, X}      # temporaries
     hashes::NTuple{2, UInt64} # hashes of time span and initial state
    epsilon::Float64           # finite difference step
end

# outer constructor
JFOp(G, x, epsilon::Float64=1e-6) =
    JFOp(G, (similar(x), similar(x)), (hash(0), hash(0)), epsilon)


# Returns true if we have previously calculated G(x, span), 
# so we can avoid recalculating it
function has_seen(op::JFOp, span, x)
    hash(span) != op.hashes[1] && return false
    hash(x)    != op.hashes[2] && return false
    return true
end

function (op::JFOp)(xy::Flows.Coupled{2}, span::NTuple{2, Real})
    # unpack
    x, y = xy[1], xy[2]

    # aliases
    xT, xTp = op.tmps

    # perturbed calculation
    xTp .= x .+ op.epsilon.*y#./norm(y).*norm(x)
    op.G(xTp, span)

    # if we have not calculated G(x, span) before, do the calculations
    if !has_seen(op, span, x)
        op.hashes = (hash(span), hash(x))
        op.G(x, span)
        xT .= x
    end

    # calc finite difference
    y .= (xTp .- xT)./op.epsilon

    return xy
end