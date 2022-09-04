module RxInferHelpersTest

using Test
using RxInfer

@testset "NamedTuple helpers" begin
    import RxInfer: fields, nthasfield

    @test fields((x = 1, y = 2)) === (:x, :y)
    @test fields((x = 1, y = 2, c = 3)) === (:x, :y, :c)
    @test fields(typeof((x = 1, y = 2))) === (:x, :y)
    @test fields(typeof((x = 1, y = 2, c = 3))) === (:x, :y, :c)

    @test nthasfield(:x, (x = 1, y = 2)) === true
    @test nthasfield(:c, (x = 1, y = 2)) === false
    @test nthasfield(:x, typeof((x = 1, y = 2))) === true
    @test nthasfield(:c, typeof((x = 1, y = 2))) === false
end

@testset "Tuple helpers" begin 
    import RxInfer: as_tuple

    @test as_tuple(1) === (1, )
    @test as_tuple((1, )) === (1, )

    @test as_tuple("string") === ("string", )
    @test as_tuple(("string", )) === ("string", )
end

@testset "Val helpers" begin 
    import RxInfer: unval

    @test unval(Val(1)) === 1
    @test unval(Val(())) === ()
    @test unval(Val(nothing)) === nothing

    @test_throws ErrorException unval(1)
    @test_throws ErrorException unval(())
    @test_throws ErrorException unval(nothing)
end

end