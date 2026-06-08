# Build the NKSearch documentation.
#
# NKSearch and its solver dependencies (Flows, GMRES) are not registered, so we
# bootstrap the docs environment here. Each dependency is taken from a local
# sibling clone when one exists (the usual development layout), otherwise it is
# fetched from its repository — so the same `julia --project=docs docs/make.jl`
# builds both locally and on CI.
import Pkg
let pkgroot = dirname(@__DIR__), parent = dirname(dirname(@__DIR__))
    specs = [Pkg.PackageSpec(path=pkgroot)]                       # NKSearch itself
    for (pkg, url) in (("Flows.jl", "https://github.com/Davide-Lasagna-s-Lab/Flows.jl"),
                       ("GMRES.jl", "https://github.com/Davide-Lasagna-s-Lab/GMRES.jl"))
        p = joinpath(parent, pkg)
        push!(specs, isdir(p) ? Pkg.PackageSpec(path=p) : Pkg.PackageSpec(url=url))
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
    format = Documenter.HTML(;
        canonical = "https://Davide-Lasagna-s-Lab.github.io/NKSearch.jl",
        # docs/src/assets/logo.svg is picked up automatically
        assets = String[],
    ),
    pages = [
        "Home"           => "index.md",
        "Concepts"       => "concepts.md",
        "Tutorial"       => "tutorial.md",
        "Solver methods" => "methods.md",
        "API reference"  => "api.md",
    ],
)

# Deploy to GitHub Pages (the gh-pages branch). Runs only in the CI job, where
# GITHUB_TOKEN / DOCUMENTER_KEY is available; a no-op on local builds.
deploydocs(;
    repo      = "github.com/Davide-Lasagna-s-Lab/NKSearch.jl",
    devbranch = "master",
    push_preview = true,
)
