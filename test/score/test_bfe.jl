module RxInferScoreTest

using Test, Random
using RxInfer

import RxInfer: get_skip_strategy, get_scheduler, apply_diagnostic_check
import ReactiveMP: InfCountingReal, FactorNodeCreationOptions, make_node, activate!

@testset "BetheFreeEnergy score tests" begin
    @testset "Diagnostic check tests" begin
        @testset "`BetheFreeEnergyCheckInfs` diagnostic" begin
            stream = Subject(Any)

            vbenergy = stream |> map(Float64, entropy)
            vbenergy = apply_diagnostic_check(BetheFreeEnergyCheckInfs(), randomvar(:x), vbenergy)

            events = []

            subscription = subscribe!(
                vbenergy |> safe(), lambda(on_next     = (data) -> push!(events, float(data)), on_error    = (err) -> push!(events, err), on_complete = () -> push!(events, "completed"))
            )

            # First value is ok
            next!(stream, NormalMeanVariance(0.0, 1.0))
            @test events[1] ≈ 1.4189385332046727

            # Second valus is NaN, but the corresponding check is not enabled
            next!(stream, NormalMeanPrecision(NaN, NaN))
            @test events[2] |> isnan

            # Third value is Inf, should trigger the check
            next!(stream, NormalMeanPrecision(0.0, 0.0))
            @test events[3] isa String && occursin("The result is `Inf`", events[3])

            # Normally stream should unsubscribe after first error 
            next!(stream, NormalMeanPrecision(0.0, 0.0))
            @test length(events) === 3
        end

        @testset "`BetheFreeEnergyCheckNaNs` for variable bound energy" begin
            stream = Subject(Any)

            vbenergy = stream |> map(Float64, entropy)
            vbenergy = apply_diagnostic_check(BetheFreeEnergyCheckNaNs(), randomvar(:x), vbenergy)

            events = []

            subscription = subscribe!(
                vbenergy |> safe(), lambda(on_next     = (data) -> push!(events, float(data)), on_error    = (err) -> push!(events, err), on_complete = () -> push!(events, "completed"))
            )

            # First value is ok
            next!(stream, NormalMeanVariance(0.0, 1.0))
            @test events[1] ≈ 1.4189385332046727

            # Second valus is Inf, but the corresponding check is not enabled
            next!(stream, NormalMeanPrecision(0.0, 0.0))
            @test events[2] |> isinf

            # Third value is NaN, should trigger the check
            next!(stream, NormalMeanPrecision(NaN, NaN))
            @test events[3] isa String && occursin("The result is `NaN`", events[3])

            # Normally stream should unsubscribe after first error 
            next!(stream, NormalMeanPrecision(0.0, 0.0))
            @test length(events) === 3
        end

        @testset "Empty diagnostic check" begin
            stream = Subject(Any)

            vbenergy = stream |> map(Float64, entropy)
            vbenergy = apply_diagnostic_check(nothing, randomvar(:x), vbenergy)

            events = []

            subscription = subscribe!(
                vbenergy |> safe(), lambda(on_next     = (data) -> push!(events, float(data)), on_error    = (err) -> push!(events, err), on_complete = () -> push!(events, "completed"))
            )

            # First value is ok
            next!(stream, NormalMeanVariance(0.0, 1.0))
            @test events[1] ≈ 1.4189385332046727

            # Second valus is Inf, but the corresponding check is not enabled
            next!(stream, NormalMeanPrecision(0.0, 0.0))
            @test events[2] |> isinf

            # Third value is NaN, but the corresponding check is not enabled
            next!(stream, NormalMeanPrecision(NaN, NaN))
            @test events[3] |> isnan

            # Normally stream should not unsubscribe if there are no errors
            next!(stream, NormalMeanPrecision(0.0, 0.0))
            @test length(events) === 4
        end
    end
end

end
