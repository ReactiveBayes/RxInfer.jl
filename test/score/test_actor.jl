module RxInferScoreActorTest

using Test, Random
using RxInfer

import RxInfer: ScoreActor, score_snapshot, score_snapshot_final, score_snapshot_iterations
import Rocket: release!

@testset "ScoreActor tests" begin

    @testset "Basic functionality #1" begin 

        actor = ScoreActor(Float64, 10, 1)

        for i in 1:10
            next!(actor, convert(Float64, i))
        end

        release!(actor)

        raw   = score_snapshot(actor)
        final = score_snapshot_final(actor)
        aggregated = score_snapshot_iterations(actor)

        @test length(raw) === 10
        @test raw == 1:10
        @test length(final) === 1
        @test final[1] == 10.0
        @test length(aggregated) === 10
        @test aggregated == 1:10

        @test_throws AssertionError release!(actor) # twice release is not allowed

        actor = ScoreActor(Float64, 10, 1)

        for i in 1:10
            next!(actor, convert(Float64, i + 1))
        end

        release!(actor)

        raw   = score_snapshot(actor)
        final = score_snapshot_final(actor)
        aggregated = score_snapshot_iterations(actor)

        @test length(raw) === 10
        @test raw == ((1:10) .+ 1)
        @test length(final) === 1
        @test final[1] == 11.0
        @test length(aggregated) === 10
        @test aggregated == ((1:10) .+ 1)

        actor = ScoreActor(Float64, 10, 1)

        for i in 1:5
            next!(actor, 0.0)
        end

        @test_logs (:warn, r"Invalid `release!`x*") release!(actor)

    end

    @testset "Basic functionality #2" begin 

        actor = ScoreActor(Float64, 10, 2)

        for i in 1:10
            next!(actor, convert(Float64, i))
        end

        release!(actor)

        raw   = score_snapshot(actor)
        final = score_snapshot_final(actor)
        aggregated = score_snapshot_iterations(actor)

        @test length(raw) === 10
        @test raw == 1:10
        @test length(final) === 1
        @test final[1] == 10.0
        @test length(aggregated) === 10
        @test aggregated == 1:10

        # Partial fe should not affect snapshots
        for i in 1:5
            next!(actor, convert(Float64, i))
        end

        raw   = score_snapshot(actor)
        final = score_snapshot_final(actor)
        aggregated = score_snapshot_iterations(actor)

        @test length(raw) === 10
        @test raw == 1:10
        @test length(final) === 1
        @test final[1] == 10.0
        @test length(aggregated) === 10
        @test aggregated == 1:10

        actor = ScoreActor(Float64, 10, 2)

        for i in 1:10
            next!(actor, convert(Float64, i + 1))
        end

        release!(actor)

        raw   = score_snapshot(actor)
        final = score_snapshot_final(actor)
        aggregated = score_snapshot_iterations(actor)

        @test length(raw) === 10
        @test raw == ((1:10) .+ 1)
        @test length(final) === 1
        @test final[1] == 11.0
        @test length(aggregated) === 10
        @test aggregated == ((1:10) .+ 1)

        # Partial fe should not affect snapshots
        for i in 1:5
            next!(actor, convert(Float64, i))
        end

        raw   = score_snapshot(actor)
        final = score_snapshot_final(actor)
        aggregated = score_snapshot_iterations(actor)

        @test length(raw) === 10
        @test raw == ((1:10) .+ 1)
        @test length(final) === 1
        @test final[1] == 11.0
        @test length(aggregated) === 10
        @test aggregated == ((1:10) .+ 1)

        actor = ScoreActor(Float64, 10, 2)

        for i in 1:5
            next!(actor, 0.0)
        end

        @test_logs (:warn, r"Invalid `release!`x*") release!(actor)

    end

    @testset "Basic functionality #2" begin 

        actor = ScoreActor(Float64, 10, 2)

        for i in 1:30
            next!(actor, convert(Float64, i))
            if rem(i, 10) === 0
                release!(actor)
            end
        end

        raw = score_snapshot(actor)
        final = score_snapshot_final(actor)
        aggregated = score_snapshot_iterations(actor)

        @test length(raw) === 20
        @test raw == 11:30
        @test length(final) == 2
        @test final == [ 20, 30 ]
        @test length(aggregated) === 10
        @test aggregated == ((11:20) .+ (21:30)) ./ 2

        # here partial application may affect as it overwrites some storage
        for i in 1:5
            next!(actor, convert(Float64, i))
        end

        raw = score_snapshot(actor)
        final = score_snapshot_final(actor)
        aggregated = score_snapshot_iterations(actor)

        @test length(raw) === 10
        @test raw == 21:30
        @test length(final) == 1
        @test final == [ 30 ]
        @test length(aggregated) === 10
        @test aggregated == 21:30

    end

end

end