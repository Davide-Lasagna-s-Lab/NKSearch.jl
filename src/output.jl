# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

# Display headers
const _header = "+------+----------+-----------+-----------+----------+-----------+----------+----------+----------+\n"*
                "| iter |   |δx|   |    δT     |    δs     |    T     |     s     |    |e|   |     λ    |    res   |\n"*
                "+------+----------+-----------+-----------+----------+-----------+----------+----------+----------+\n"

display_header(io::IO)= (print(io, _header); flush(io))

function display_status(io::IO, iter, δx_norm, δT, δs, T, s, e_norm, λ, res_err_norm)
    str = @sprintf "|%4d  | %5.2e | %+5.2e | %+5.2e | %5.2e | %+5.2e | %5.2e | %5.2e | %5.2e |" iter δx_norm δT δs T s e_norm λ res_err_norm
    println(io, str)
    flush(io) 
end