using Test
using NKSearch 

@testset "zero-th order                          " begin
    # define object
    sc = SolCache([1.0], 0)

    # test
    out = [0.0]
    NKSearch.dds!(sc, out)
    @test abs(out[1] - 0.0) < 1e-16
end

@testset "first order interpolation              " begin
    # push values of a known function
    fun(s)  = 2*s + 1
    funp(s) = 2

    # define object
    sc = SolCache([fun(0.0)], 0)

    # add one bit
    push!(sc, [fun(1.0)], 1)

    # check third order interpolation
    out = [0.0]
    NKSearch.dds!(sc, out)
    @test abs(out[1] - funp(1)) < 1e-16
end

@testset "second order interpolation             " begin
    # push values of a known function
    fun(s)  = 2*s^2 + 3*s + 1
    funp(s) = 4*s   + 3

    # define object
    sc = SolCache([fun(0.0)], 0)

    # add two bits
    push!(sc, [fun(1.0)],  1)
    push!(sc, [fun(2.5)], 2.5)

    # check third order interpolation
    out = [0.0]
    NKSearch.dds!(sc, out)
    @test abs(out[1] - funp(2.5)) < 1e-14
end

@testset "third order interpolation              " begin
    # push values of a known function
    fun(s)  = 2*s^3 + 3*s^2 + 1*s - 1
    funp(s) = 6*s^2 + 6*s   + 1

    # define object
    sc = SolCache([fun(0.0)], 0)

    # add two bits
    push!(sc, [fun(1.0)], 1.0)
    push!(sc, [fun(2.5)], 2.5)
    push!(sc, [fun(3.1)], 3.1)

    # check third order interpolation
    out = [0.0]
    NKSearch.dds!(sc, out)
    @test abs(out[1] - funp(3.1)) < 1e-14

    # test increasing arclength
    @test_throws ArgumentError push!(sc, [fun(3.1)], 3.1)

    # push further
    push!(sc, [fun(3.2)], 3.2)
    out = [0.0]
    NKSearch.dds!(sc, out)
    @test abs(out[1] - funp(3.2)) < 1e-13
end