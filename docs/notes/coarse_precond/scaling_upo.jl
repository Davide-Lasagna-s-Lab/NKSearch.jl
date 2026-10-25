# Newton searches for long KS periodic orbits with :ls_iterative, with and
# without the coarse preconditioner. A long orbit is the M-fold repetition of a
# converged base orbit (a periodic orbit of period M T0 whose Floquet
# multipliers are the M-th powers of those of the base orbit). Its seeds are
# perturbed by a relative amount `pert`, and Newton reconverges it. We record
# the GMRES iterations of every Newton step.
# Usage: julia -t 2 scaling_upo.jl upos/T_xx.txt
include(joinpath(@__DIR__, "ks.jl"))
using NKSearch, Random, Printf

lines = readlines(ARGS[1])
T0 = parse(Float64, lines[1])
base = [parse.(Float64, split(l)) for l in lines[2:end]]
N0 = length(base)
const Ms = parse.(Int, split(get(ENV, "MS", "1,2,4,8,16,32"), ","))
const ks_ = parse.(Int, split(get(ENV, "KS", "0,8"), ","))
const pert = parse(Float64, get(ENV, "PERT", "1e-3"))

G, L, F = ks_flows()
@printf "base orbit T0 = %.4f, %d segments of length %.3f\n" T0 N0 T0/N0
@printf "%4s %4s %8s %4s %7s %9s  %s\n" "M" "N" "T" "k" "Newton" "residual" "GMRES iterations per Newton step"
for M in Ms, k in ks_
    N = M*N0
    Random.seed!(4)
    z = MVector(ntuple(i->base[(i-1)%N0+1] .* (1 .+ pert.*randn(32)), N), M*T0*(1 + pert*randn()))
    its = Int[]
    opts = Options(method=:ls_iterative, coarse_dim=k, coarse_sweeps=5, maxiter=20,
                   e_norm_tol=1e-10, dz_norm_tol=1e-12, gmres_rtol=1e-3, gmres_maxiter=3000,
                   gmres_verbose=false, verbose=false,
                   gmres_callback=(it, args...)->(it == 0 ? push!(its, 0) : (its[end] = it); false))
    t = @elapsed search!(G, L, F, z, opts)
    e = NKSearch.e_norm_λ(ntuple(i->deepcopy(G), N), nothing, z, z, 0.0, ntuple(i->similar(z[1]), N))
    @printf "%4d %4d %8.1f %4d %7d %9.1e  %s   (%.0f s)\n" M N M*T0 k length(its) e join(its, " ") t
    flush(stdout)
end
