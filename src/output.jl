# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #
using Printf

# ~~~ HEADERS ~~~

# line search
const _header_1_ls = "+------+----------+-----------+----------+-----------+----------+\n"*
                     "| iter |   |dz|   |     T     |   ||e||  |     λ     |    res   |\n"*
                     "+------+----------+-----------+----------+-----------+----------+\n"

const _header_2_ls = "+------+----------+-----------+-----------+----------+----------+----------+\n"*
                     "| iter |  ||dz||  |    T      |     s     |   ||e||  |     λ    |    res   |\n"*
                     "+------+----------+-----------+-----------+----------+----------+----------+\n"

const _headers_ls = [_header_1_ls, _header_2_ls]

display_header_ls(io::IO, ::MVector{X, N, NS}) where {X, N, NS} = 
    (print(io, _headers_ls[NS]); flush(io))

# trust region
const _header_tr = "+------+--------+-----------+-----------+------------+-----------+-----------+\n"*
                   "| iter | which  |   step    |   ||e||   |    rho     | tr_radius |  ||dz||   |\n"*
                   "+------+--------+-----------+-----------+------------+-----------+-----------+\n"

display_header_tr(io::IO, ::MVector{X, N, NS}) where {X, N, NS} = 
    (print(_header_tr); flush(io))

# ~~~ DISPLAY FUNCTIONS ~~~

# line search

# print output when we have a shift
function display_status_ls(io::IO, iter, dz_norm, d::TUP2, e_norm, λ, res_err_norm) where {TUP2<:Tuple{Any, Any}}
    str = @sprintf "|%4d  | %5.2e | %+5.2e | %+5.2e | %5.2e | %+5.2e | %5.2e |" iter dz_norm d[1] d[2] e_norm λ res_err_norm
    println(io, str)
    flush(io)
    return nothing
end

# print output when we don't
function display_status_ls(io::IO, iter, dz_norm, d::Tuple{Any}, e_norm, λ, res_err_norm)
    str = @sprintf "|%4d  | %5.2e | %+5.2e | %5.2e | %5.2e | %5.2e |" iter dz_norm d[1] e_norm λ res_err_norm
    println(io, str)
    flush(io)
    return nothing
end

# trust region
function display_status_tr(io::IO, iter, which, step, e_norm, rho, tr_radius, dz_norm)
    str = @sprintf "|%4d  | %s | %5.3e | %5.3e | %+5.3e | %5.3e | %5.3e |" iter lpad(which, 6) step e_norm rho tr_radius dz_norm
    println(str)
    flush(stdout)
end
