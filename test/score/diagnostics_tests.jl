@testitem "`ObjectiveDiagnosticCheckInfs` diagnostic" begin
    import RxInfer: ObjectiveDiagnosticCheckInfs, apply_diagnostic_check

    stream = Subject(Any)

    vbenergy = stream |> map(Float64, entropy)
    vbenergy = apply_diagnostic_check(ObjectiveDiagnosticCheckInfs(), nothing, vbenergy)

    events = []

    subscription = subscribe!(
        vbenergy |> safe(), lambda(on_next = (data) -> push!(events, float(data)), on_error = (err) -> push!(events, err), on_complete = () -> push!(events, "completed"))
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

@testitem "`ObjectiveDiagnosticCheckNaNs` diagnostics" begin
    import RxInfer: ObjectiveDiagnosticCheckNaNs, apply_diagnostic_check

    stream = Subject(Any)

    vbenergy = stream |> map(Float64, entropy)
    vbenergy = apply_diagnostic_check(ObjectiveDiagnosticCheckNaNs(), nothing, vbenergy)

    events = []

    subscription = subscribe!(
        vbenergy |> safe(), lambda(on_next = (data) -> push!(events, float(data)), on_error = (err) -> push!(events, err), on_complete = () -> push!(events, "completed"))
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

@testitem "Empty diagnostic check" begin
    import RxInfer: apply_diagnostic_check

    stream = Subject(Any)

    vbenergy = stream |> map(Float64, entropy)
    vbenergy = apply_diagnostic_check(nothing, nothing, vbenergy)

    events = []

    subscription = subscribe!(
        vbenergy |> safe(), lambda(on_next = (data) -> push!(events, float(data)), on_error = (err) -> push!(events, err), on_complete = () -> push!(events, "completed"))
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
