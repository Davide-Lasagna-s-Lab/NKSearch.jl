@testset "MVector - interface                    " begin

    a = MVector(([1, 2, 3], [4, 5, 6], [7, 8, 9]), 2.0, 4.0)
    b = copy(a)
    @test a[1] == [1, 2, 3]
    @test a[2] == [4, 5, 6]
    @test a[3] == [7, 8, 9]
    @test a.d  == (2.0, 4.0)
    @test b[1] == [1, 2, 3]
    @test b[2] == [4, 5, 6]
    @test b[3] == [7, 8, 9]
    @test b.d  == (2.0, 4.0)

    c = tovector(a)
    @test c == [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 4]
    d = fromvector!(similar(a), c)
    @test d[1] == [1, 2, 3]
    @test d[2] == [4, 5, 6]
    @test d[3] == [7, 8, 9]
    @test d.d  == (2.0, 4.0)
end

@testset "MVector - dot and norm                 " begin
    # single seed
    a = MVector(([1, 2, 3], [4, 5, 6], [7, 8, 9]), 2.0, 4.0)
    b = MVector(([1, 2, 5], [4, 5, 7], [7, 8, 1]), 3.0, 1.0)

    @test norm(a)   == norm([1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 4])
    @test dot(a, b) ==  dot([1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 4],
                            [1, 2, 5, 4, 5, 7, 7, 8, 1, 3, 1])

    # test dot product does not allocate
    alloc(a, b) = @allocated dot(a, b)
    @test alloc(a, b) == 0
end

@testset "MVector - broadcast                    " begin
    a = MVector(([1, 2, 3], [4, 5, 6], [7, 8, 9]), 1.0, 4.0)
    b = MVector(([1, 2, 5], [4, 5, 7], [7, 8, 1]), 3.0, 1.0)

    a .= a .+ b
    @test a.x[1] == [2, 4, 8]
    @test a.x[2] == [8, 10, 13]
    @test a.x[3] == [14, 16, 10]
    @test a.d    == (4, 5)
end

@testset "MVector - broadcast allocation         " begin
    a = MVector(([1, 2, 3], [4, 5, 6], [7, 8, 9]), 1.0, 4.0)
    b = MVector(([1, 2, 5], [4, 5, 7], [7, 8, 1]), 3.0, 1.0)
    c = 1
    d = 1
    alloc(a, b, c, d) = (@allocated a .+= b .* 2.0 .+ 3.0.*a)
    @test alloc(a, b, c, d) == 0
end

@testset "MVector - io                           " begin
    a = MVector(([1, 2, 3], [4, 5, 6], [7, 8, 9]), 1.0, 4.0)
    dt = 2.0
    
    # first save
    save_seed(a, "test.file", Dict("dt"=>dt))
    
    # then load
    b, dict = load_seed!(similar(a), "test.file")

    @test b[1] == [1, 2, 3]
    @test b[2] == [4, 5, 6]
    @test b[3] == [7, 8, 9]
    @test b.d == (1.0, 4.0)
    @test length(keys(dict)) == 1
    @test dict["dt"] == 2.0

    # check for mistakes
    c = MVector(([1, 2, 3],), 1.0, 4.0)
    @test_throws ArgumentError load_seed!(c, "test.file")

    d = MVector(([1, 2, 3], [4, 5, 6], [7, 8, 9]), 1.0)
    @test_throws ArgumentError load_seed!(c, "test.file")

    # with complex input
    e = MVector(([1+im, 2+2*im, 3+3*im],), 1.0)
    save_seed(e, "test2.file")
    f, dict = load_seed!(similar(e), "test2.file")
    @test f[1] == e[1]
    @test f.d  == e.d
end