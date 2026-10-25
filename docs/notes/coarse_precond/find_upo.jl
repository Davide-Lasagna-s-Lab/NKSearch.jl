# Find periodic orbits of KS (L = 39, odd subspace) from near recurrences of a
# chaotic trajectory, with the hookstep Newton-Krylov method of NKSearch
# (no coarse preconditioner). Converged orbits are written to upos/T_<T>.txt:
# one line per seed (segments of length ≈ τ), the period on the first line.
include(joinpath(@__DIR__, "ks.jl"))
using NKSearch, Random, Printf

const τ = 2.0
const outdir = joinpath(@__DIR__, "upos"); mkpath(outdir)

G, L, F = ks_flows()
Random.seed!(3)
a = 0.1*randn(32); G(a, (0, 300.0))

# trajectory sampled every Δ
Δ, nsamp = 0.25, 8000
traj = [copy(a)]
for n in 2:nsamp
    G(a, (0, Δ)); push!(traj, copy(a))
end

# near recurrences: local minima of |a(t+T) - a(t)| / |a(t)| for T in [15, 80]
cands = Tuple{Float64, Int, Int}[]
for m in round(Int, 15/Δ):round(Int, 80/Δ), n in 1:20:nsamp-m
    push!(cands, (norm(traj[n+m] .- traj[n])/norm(traj[n]), n, m))
end
sort!(cands)
picked = Tuple{Float64, Int, Int}[]
for c in cands
    all(p->abs(p[2]-c[2]) > 200 || abs(p[3]-c[3]) > 8, picked) && push!(picked, c)
    length(picked) == 25 && break
end

for (r, n, m) in picked
    T = m*Δ
    N = max(2, round(Int, T/τ))
    x = copy(traj[n]); seeds = [copy(x)]
    for i in 2:N
        G(x, (0, T/N)); push!(seeds, copy(x))
    end
    z = MVector(Tuple(seeds), T)
    @printf "candidate T = %6.2f, recurrence %.3f, N = %d\n" T r N
    status = search!(G, L, F, z, Options(method=:tr_iterative, maxiter=40, e_norm_tol=1e-10,
                                         gmres_maxiter=200, gmres_rtol=1e-4, gmres_verbose=false,
                                         verbose=false, tr_radius_init=0.1, min_step=1e-8))
    tmps = ntuple(i->similar(z[1]), N)
    e = NKSearch.e_norm_λ(ntuple(i->deepcopy(G), N), nothing, z, z, 0.0, tmps)
    @printf "    -> %s, residual %.2e, T = %.4f\n" status e z.d[1]
    if e < 1e-9
        open(joinpath(outdir, @sprintf("T_%.3f.txt", z.d[1])), "w") do io
            println(io, z.d[1])
            for s in z.x
                println(io, join(s, " "))
            end
        end
    end
    flush(stdout)
end
