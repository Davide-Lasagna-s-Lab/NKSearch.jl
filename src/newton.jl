# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

import GMRES

export search!

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

search!(G, L, S, F, dS, z0::MVector{X, N, 2}, opts::Options=Options()) where {X, N} =
    _search!(G, L, S, (F, dS), z0, opts)

# when we do not have shifts
search!(G, L, F, z0::MVector{X, N, 1}, opts::Options=Options()) where {X, N} =
    _search!(G, L, nothing, (F, ), z0, opts)

# dispatch to correct method
function _search!(G, L, S, D, z0::MVector{X, N, NS}, opts) where {X, N, NS}
    if    opts.method == :linesearch
        A = MMatrix(G, L, S, D, z0, opts)
        return _search_linesearch!(G, L, S, D, z0, A, opts)
    elseif opts.method == :hookstep
        return _search_hookstep!(G, L, S, D, z0, opts)
    else
        throw(ArgumentError("invalid method"))
    end
end