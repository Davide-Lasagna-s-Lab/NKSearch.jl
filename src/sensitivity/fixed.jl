# ~~~ Period and shift perturbations held fixed in the sensitivity problems ~~~
#
# The sensitivity problems border the shooting operator with one column and one phase row per
# entry of `D`: the time derivative D[1](x) = F(x) and the generators D[2:end] of the continuous
# shifts. The bordered operator is non-singular only if these directions are linearly independent
# at the start of the orbit. For a travelling wave, F(x) = -c ∂ₓx is parallel to the shift
# generator, and for an equilibrium F(x) = 0: the period is then arbitrary (any T, with s = cT,
# closes the orbit) and the operator has an exact null vector. Holding the perturbation of the
# period fixed removes it. Index i of `fixed` drops column i, replaces phase row i by the
# identity and sets entry i of the right-hand side to zero, so that the solution has δd[i] = 0.

export degenerate_directions

"""
    degenerate_directions(D, x; rtol=1e-3) -> NTuple{K, Int}

The indices `i` for which the direction `D[i](out, x)` is linearly dependent on the other
directions of `D` at the state `x`, where `D` holds the time-derivative and shift-generator
operators of [`search!`](@ref) and of the sensitivity problems.

The directions are examined from the last to the first, so that the time direction `D[1]` is the
one reported in favour of the shift generators. Direction `i` is dependent if

- its norm is at most `rtol` times the largest norm of the directions (it vanishes, as the time
  derivative of an equilibrium does), or
- its component orthogonal to the directions kept so far has norm at most `rtol` times its own
  norm (it is parallel to them, as the time derivative of a travelling wave is to the shift
  generator).

An empty tuple means the directions are independent and the bordered sensitivity problems are
well posed. Otherwise pass the result as the `fixed` keyword of `make_tangent_problem` and
[`make_adjoint_problem`](@ref), or pass `fixed=:auto`.
"""
function degenerate_directions(D, x; rtol::Real=1e-3)
    vs = map(Di -> (v = similar(x); Di(v, x); v), Tuple(D))
    norms = map(norm, vs)
    vmax = maximum(norms)

    kept = typeof(x)[]    # orthonormal basis of the directions kept so far
    dependent = Int[]
    for i in reverse(eachindex(vs))
        if norms[i] <= rtol*vmax
            push!(dependent, i)
            continue
        end
        r = copy(vs[i])
        for _ in 1:2, q in kept    # Gram-Schmidt, with one re-orthogonalisation pass
            r .-= dot(q, r).*q
        end
        nr = norm(r)
        if nr <= rtol*norms[i]
            push!(dependent, i)
        else
            r .*= 1/nr
            push!(kept, r)
        end
    end
    return Tuple(sort!(dependent))
end

# The indices of the orbit unknowns `z.d` held fixed in a sensitivity problem with `NS` such
# unknowns and period `T`, from the `fixed` keyword: `:auto` uses `degenerate_directions`, a tuple
# is validated and used as given. With the default `()` nothing is held fixed, and a warning is
# logged if the directions of `D` are dependent at `x0`, since the problem is then singular.
function _resolve_fixed(fixed, D, x0, NS::Integer, T::Real, rtol::Real)
    if fixed === :auto
        return degenerate_directions(D, x0; rtol)
    end
    fixed isa Tuple ||
        throw(ArgumentError("`fixed` must be a tuple of indices into the period and shifts, or :auto; got $(repr(fixed))"))
    all(i -> i isa Integer && 1 <= i <= NS, fixed) ||
        throw(ArgumentError("`fixed` must hold indices between 1 and $NS (the period and shifts of the orbit); got $(repr(fixed))"))
    allunique(fixed) ||
        throw(ArgumentError("`fixed` must not repeat an index; got $(repr(fixed))"))
    if isempty(fixed)
        dependent = degenerate_directions(D, x0; rtol)
        isempty(dependent) ||
            @warn "the period/shift directions D$(collect(dependent)) are linearly dependent at the start of the orbit (travelling wave or equilibrium), so the sensitivity problem is singular and GMRES may stall above its tolerance; pass `fixed=$(dependent)`, or `fixed=:auto`, to remove the dependent directions" dependent period=T
    end
    return map(Int, fixed)
end
