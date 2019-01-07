# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

# Display headers
const _header_1 = "+------+----------+-----------+----------+----------+----------+----------+\n"*
                  "| iter |   |δx|   |    δT     |    T     |    |e|   |     λ    |    res   |\n"*
                  "+------+----------+-----------+----------+----------+----------+----------+\n"

const _header_2 = "+------+----------+-----------+-----------+----------+-----------+----------+----------+----------+\n"*
                  "| iter |   |δx|   |    δT     |    δs     |    T     |     s     |    |e|   |     λ    |    res   |\n"*
                  "+------+----------+-----------+-----------+----------+-----------+----------+----------+----------+\n"

const _headers = [_header_1, _header_2]

display_header(io::IO, ::MVector{X, N, NS})= (print(io, _headers[NS]); flush(io))

# print output when we have a shift
function display_status(io::IO, iter, δx_norm, δd::TUP2, d::TUP2, e_norm, λ, res_err_norm) where {TUP2<:Tuple{Any, Any}}
    str = @sprintf "|%4d  | %5.2e | %+5.2e | %+5.2e | %5.2e | %+5.2e | %5.2e | %5.2e | %5.2e |" iter δx_norm δd[1] δd[2] d[1] d[2] e_norm λ res_err_norm
    println(io, str)
    flush(io) 
    return nothing
end

# print output when we don't
function display_status(io::IO, iter, δx_norm, δT, T, e_norm, λ, res_err_norm) where {TUP2<:Tuple{Any, Any}}
    str = @sprintf "|%4d  | %5.2e | %+5.2e | %5.2e | %5.2e | %5.2e | %5.2e |" iter δx_norm δT T e_norm λ res_err_norm
    println(io, str)
    flush(io) 
    return nothing
end