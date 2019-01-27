# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

export JFOp

# Object to approximate the action of the linearised flow operator
# using a finite difference quotient with nonlinear simulations
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
    xTp .= x .+ op.epsilon.*y
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