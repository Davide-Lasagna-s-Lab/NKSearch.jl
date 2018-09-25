# ----------------------------------------------------------------- #
# Copyright 2017-18, Davide Lasagna, AFM, University of Southampton #
# ----------------------------------------------------------------- #

using Base.Test
using asis

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
@testset "MVector - interface                   " begin
    # test constructor
    @test_throws ArgumentError MVector(([1, 2, 3], [7, 8, 9]), (0.0, 1.0, 2.0), 4.0)

    a = MVector(([1, 2, 3], [4, 5, 6], [7, 8, 9]), (0.0, 1.0, 2.0), 4.0)
    b = copy(a)
    @test a[1] == [1, 2, 3]
    @test a[2] == [4, 5, 6]
    @test a[3] == [7, 8, 9]
    @test a.T  == (0.0, 1.0, 2.0)
    @test a.s  == 4.0    
    @test b[1] == [1, 2, 3]
    @test b[2] == [4, 5, 6]
    @test b[3] == [7, 8, 9]
    @test b.T  == (0.0, 1.0, 2.0)
    @test b.s  == 4.0
end

@testset "MVector - dot and norm                " begin
    # single seed
    a = MVector(([1, 2, 3], [4, 5, 6], [7, 8, 9]), (0.0, 1.0, 2.0), 4.0)
    b = MVector(([1, 2, 5], [4, 5, 7], [7, 8, 1]), (0.0, 3.0, 2.0), 1.0)

    @test norm(a)   == norm([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 4])
    @test dot(a, b) ==  dot([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 4],
                            [1, 2, 5, 4, 5, 7, 7, 8, 1, 0, 3, 2, 1])
    
    # test dot product does not allocate
    alloc(a, b) = @allocated dot(a, b)
    @test alloc(a, b) == 0
end

@testset "MVector - broadcast                   " begin
    a = MVector(([1, 2, 3], [4, 5, 6], [7, 8, 9]), (0.0, 1.0, 2.0), 4.0)
    b = MVector(([1, 2, 5], [4, 5, 7], [7, 8, 1]), (0.0, 3.0, 2.0), 1.0)

    a .= a .+ b
    @test a.x[1] == [2, 4, 8]
    @test a.x[2] == [8, 10, 13]
    @test a.x[3] == [14, 16, 10]
    @test a.T    == (0, 4, 4)
    @test a.s    ==  5
end

@testset "MVector - broadcast allocation        " begin
    a = MVector(([1, 2, 3], [4, 5, 6], [7, 8, 9]), (0.0, 1.0, 2.0), 4.0)
    b = MVector(([1, 2, 5], [4, 5, 7], [7, 8, 1]), (0.0, 3.0, 2.0), 1.0)
    c = 1
    alloc(a, b, c) = (@allocated a .+= b .* c)
    @test alloc(a, b, c) == 0
end