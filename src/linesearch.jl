# -------------------------------------------------------------- #
# Copyright 2017, Davide Lasagna, AFM, University of Southampton #
# -------------------------------------------------------------- #

function e_norm_λ(G, S, z0::MVector{X, N}, δz::MVector{X, N}, λ::Real, tmp::X) where {X, N}
    # initialize
    val_λ = zero(norm(tmp)^2)
    
    # sum all contributions
    for i = 1:N
        tmp  .= z0[i] .+ λ.*δz[i]
        Ti    = i == 1 ? z0.T[1] + λ*δz.T[1] :
                i == N ? z0.T[3] + λ*δz.T[3] : z0.T[2] + λ*δz.T[2]/(N-2)
        G(tmp, (0, Ti))
        i == N && S(tmp, z0.s + λ*δz.s)
        tmp .-= z0[i%N+1] .+ λ.*δz[i%N+1] 
        val_λ += norm(tmp)^2
    end

    return val_λ
end



function linesearch(G, S, z0::MVector{X, N}, δz::MVector{X, N}, opts::Options, tmp::X) where {X, N}
    # current error
    val_0 = e_norm_λ(G, S, z0, δz, 0.0, tmp)

    # start with full Newton step
    λ = 1.0

    for iter = 1:opts.ls_maxiter
        # calc error
        val_λ = e_norm(G, S, z0, δz, λ, tmp)
        
        # accept any reduction of error
        val_λ < val_0 && return λ, val_λ
        
        # ~ otherwise attempt with shorter step ~
        λ *= opts.ls_rho
    end 

    error("maximum number of line search iterations reached")
end