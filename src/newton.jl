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

search!(G, L, S, F, dS, z0::MVector{X, N, 2}, opt::Options=Options()) where {X, N} =
    _search!(G, L, S, (F, dS), z0, opt)

# when we do not have shifts
search!(G, L, F, z0::MVector{X, N, 1}, opt::Options=Options()) where {X, N} =
    _search!(G, L, nothing, (F, ), z0)

# dispatch to correct method
function _search!(G, L, S, D, z0, opts)
    if opt.method == :linesearch
        return _search_ls!(G, L, S, D, z0, opts)
    elseif opt.method == :hookstep
        return _search_hs!(G, L, S, D, z0, opts)
    else
        throw(ArgumentError("method must be 'linesearch' or 'hookstep'"))
    end
end