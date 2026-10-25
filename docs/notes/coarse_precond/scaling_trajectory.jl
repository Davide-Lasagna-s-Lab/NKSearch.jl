# GMRES iterations for the Newton operator of NKSearch (:ls_iterative, Ashtari
# ordering), built along a chaotic KS trajectory with segments of fixed length
# τ, for increasing N = T/τ. The trajectory is not a periodic orbit: the
# operator is that of the multiple-shooting system closed cyclically, with the
# period column and the phase row, at seeds spaced τ apart on the attractor.
# The right-hand side is random. Run with `julia -t 2 scaling_trajectory.jl`;
# environment variables TAU, TS (periods) or NS (segment counts), KS, SWEEPS.
include(joinpath(@__DIR__, "ks.jl"))
using NKSearch, Random, Printf

const τ = parse(Float64, get(ENV, "TAU", "2.0"))
# segment counts: from the periods in TS if given (N = T/τ), otherwise from NS
const Ns = haskey(ENV, "TS") ? [round(Int, parse(Float64, t)/τ) for t in split(ENV["TS"], ",")] :
                               parse.(Int, split(get(ENV, "NS", "8,16,32,64,128,256,512"), ","))
const ks_ = parse.(Int, split(get(ENV, "KS", "0,4,8,12"), ","))

G, L, F = ks_flows()
Random.seed!(1)
a = 0.1*randn(32); G(a, (0, 300.0))
seeds = [copy(a)]
for i in 2:maximum(Ns)
    G(a, (0, τ)); push!(seeds, copy(a))
end

@printf "# tau = %.1f\n" τ
@printf "%5s %6s %4s %6s %8s %10s %8s\n" "N" "T" "k" "GMRES" "rel.res" "coarse" "time[s]"
for N in Ns, k in ks_
    z = MVector(ntuple(i->copy(seeds[i]), N), N*τ)
    its = Ref(0)
    opts = Options(method=:ls_iterative, coarse_dim=k, coarse_sweeps=parse(Int, get(ENV, "SWEEPS", "5")),
                   gmres_maxiter=3000, gmres_rtol=1e-6, gmres_verbose=false,
                   gmres_callback=(it, args...)->(its[] = it; false))
    A = NKSearch.IterSolCache(ntuple(i->deepcopy(G), N), ntuple(i->deepcopy(L), N), nothing, (F,), z, opts)
    b = similar(z)
    tc = @elapsed NKSearch.update!(A, b, z, opts)
    Random.seed!(2); foreach(randn!, b.x); b.d = (randn(),)
    ts = @elapsed x, _ = NKSearch._solve(zero(z), A, b, opts)
    r = A*x; r .-= b
    @printf "%5d %6.0f %4d %6d %8.1e %10s %8.1f\n" N N*τ k its[] norm(r)/norm(b) (k == 0 ? "-" : string(N*k+1)) ts+tc
    flush(stdout)
end
