# -------------------------------------------------------------- #
# Copyright 2017, Davide Lasagna, AFM, University of Southampton #
# -------------------------------------------------------------- #
import Flows

function e_norm_λ(G,
                  S,
                 z0::MVector{X, N, NS},
                 δz::MVector{X, N, NS},
                  λ::Real,
                tmp::X) where {X, N, NS}

    # initialize
    val_λ = zero(norm(tmp)^2)

    # sum all error contributions
    for i = 1:N
        # set initial condition
        tmp  .= z0[i] .+ λ.*δz[i]

        # and propagation time
        Ti = (z0.d[1] + λ*δz.d[1])/N

        # actual propagation
        G(tmp, (0, Ti))

        # last segment is shifted (if we have a shift)
        NS == 2 && i == N && S(tmp, z0.d[2] + λ*δz.d[2])

        # calc difference
        tmp .-= z0[i%N+1] .+ λ.*δz[i%N+1]

        # add to error
        val_λ += norm(tmp)^2
    end

    return val_λ
end



function linesearch(G, S, z0::MVector{X, N}, δz::MVector{X, N}, opts::Options, tmp::X) where {X, N}
    # current error
    val_0 = e_norm_λ(G, S, z0, δz, 0.0, tmp)

    # start with full Newton step
    λ = 1.0

    # initialize this variable
    val_λ = λ*val_0

    for iter = 1:opts.ls_maxiter
        # calculate error
        try
            val_λ = e_norm_λ(G, S, z0, δz, λ, tmp)
        catch err
            # We might end up in a situation where the
            # new time span has nehative length. In
            # such a case, we might just continue
            if !isa(err, Flows.InvalidSpanError)
                rethrow(err)
            end
        end

        # accept any reduction of error
        val_λ < val_0 && return λ, val_λ

        # ~ otherwise attempt with shorter step ~
        λ *= opts.ls_rho
    end

    error("maximum number of line search iterations reached")
end