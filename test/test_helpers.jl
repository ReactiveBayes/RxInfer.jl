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

end