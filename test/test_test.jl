@testitem "addition" begin
    @test 1 + 2 == 3
    @test 0 + 2 == 2
    @test -1 + 2 == 1
end
@testitem "multiplication" begin
    @test 1 * 2 == 2
    @test 0 * 2 == 0
    @test -1 * 2 == -2
end