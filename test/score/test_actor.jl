module RxInferScoreActorTest

using Test, Random
using RxInfer

import RxInfer: ScoreActor, score_raw, score_final_only, score_aggreate_iterations
import Rocket: release!

@testset "ScoreActor tests" begin

    @testset "Basic functionality #1" begin 

        actor = ScoreActor(Float64, 10, 1)

        for i in 1:10
            next!(actor, convert(Float64, i))
        end

        release!(actor)

        raw   = score_raw(actor)
        final = score_final_only(actor)
        aggregated = score_aggreate_iterations(actor)

        @test size(raw) === (10, 1)
        @test raw[:, 1] == 1:10
        @test length(final) === 1
        @test final[1] == 10.0
        @test length(aggregated) === 10
        @test aggregated == 1:10

        actor = ScoreActor(Float64, 10, 1)

        for i in 1:10
            next!(actor, convert(Float64, i + 1))
        end

        release!(actor)

        raw   = score_raw(actor)
        final = score_final_only(actor)
        aggregated = score_aggreate_iterations(actor)

        @test size(raw) === (10, 1)
        @test raw[:, 1] == ((1:10) .+ 1)
        @test length(final) === 1
        @test final[1] == 11.0
        @test length(aggregated) === 10
        @test aggregated == ((1:10) .+ 1)

        actor = ScoreActor(Float64, 10, 1)

        for i in 1:5
            next!(actor, 0.0)
        end

        @test_throws AssertionError release!(actor)

    end

    @testset "Basic functionality #2" begin 

        actor = ScoreActor(Float64, 10, 2)

        for i in 1:10
            next!(actor, convert(Float64, i))
        end

        release!(actor)

        raw   = score_raw(actor)
        final = score_final_only(actor)
        aggregated = score_aggreate_iterations(actor)

        @test size(raw) === (10, 1)
        @test raw[:, 1] == 1:10
        @test length(final) === 1
        @test final[1] == 10.0
        @test length(aggregated) === 10
        @test aggregated == 1:10

        actor = ScoreActor(Float64, 10, 2)

        for i in 1:10
            next!(actor, convert(Float64, i + 1))
        end

        release!(actor)

        raw   = score_raw(actor)
        final = score_final_only(actor)
        aggregated = score_aggreate_iterations(actor)

        @test size(raw) === (10, 1)
        @test raw[:, 1] == ((1:10) .+ 1)
        @test length(final) === 1
        @test final[1] == 11.0
        @test length(aggregated) === 10
        @test aggregated == ((1:10) .+ 1)

        actor = ScoreActor(Float64, 10, 2)

        for i in 1:5
            next!(actor, 0.0)
        end

        @test_throws AssertionError release!(actor)

    end

end

end