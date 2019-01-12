# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

module NKSearch

# UTILS

# global row indices of the i-th nxn block (one-based)
@inline _blockrng(i::Integer, n::Integer) = ((i-1)*n+1):(i*n)

include("options.jl")
include("multivector.jl")
include("iter_sol_cache.jl")
include("direct_sol_cache.jl")
include("output.jl")
include("linesearch.jl")
include("newton.jl")
include("search_linesearch.jl")
include("search_hookstep.jl")

end