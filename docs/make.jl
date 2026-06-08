# Build the NKSearch documentation.
#
# NKSearch and its solver dependencies (Flows, GMRES) are not registered, so we
# bootstrap the docs environment here: if sibling clones live next to this
# repository (the usual development layout), `dev` them, then instantiate. This
# makes a single `julia --project=docs docs/make.jl` build from a clean checkout.
import Pkg
let pkgroot = dirname(@__DIR__), parent = dirname(dirname(@__DIR__))
    specs = [Pkg.PackageSpec(path=pkgroot)]                       # NKSearch itself
    for pkg in ("Flows.jl", "GMRES.jl")                           # unregistered deps
        p = joinpath(parent, pkg)
        isdir(p) && push!(specs, Pkg.PackageSpec(path=p))
    end
    Pkg.develop(specs)                                            # add them together
    Pkg.instantiate()
end

using Documenter
using NKSearch

DocMeta.setdocmeta!(NKSearch, :DocTestSetup, :(using NKSearch); recursive=true)

makedocs(;
    sitename = "NKSearch.jl",
    modules  = [NKSearch],
    authors  = "Davide Lasagna",
    checkdocs = :exports,
    doctest   = true,
    pages = [
        "Home"           => "index.md",
        "Concepts"       => "concepts.md",
        "Tutorial"       => "tutorial.md",
        "Solver methods" => "methods.md",
        "API reference"  => "api.md",
    ],
)

# Uncomment and configure once a deployment target is set up.
# deploydocs(; repo = "github.com/Davide-Lasagna-s-Lab/NKSearch.jl")
