module RxInferScoreTest

using Test
using RxInfer
using Random

@testset "BetheFreeEnergy score tests" begin
    import ReactiveMP: InfCountingReal

    @testset "`BetheFreeEnergyCheckInfs` for variable bound energy" begin

        # Dummy graph [U] - (x) - [U]
        model = FactorGraphModel()
        x     = randomvar(model, :x)
        make_node(Uninformative, FactorNodeCreationOptions(), x)
        make_node(Uninformative, FactorNodeCreationOptions(), x)

        activate!(model)

        objective = BetheFreeEnergy(IncludeAll(), BetheFreeEnergyCheckInfs())

        vbenergy = score(InfCountingReal{Float64}, objective, VariableBoundEntropy(), x, AsapScheduler())

        events = []

        subscription = subscribe!(
            vbenergy |> safe(),
            lambda(
                on_next     = (data) -> push!(events, float(data)),
                on_error    = (err) -> push!(events, err),
                on_complete = () -> push!(events, "completed")
            )
        )

        # First value is ok
        setmarginal!(x, NormalMeanVariance(0.0, 1.0))
        @test events[1] ≈ 1.4189385332046727

        # Second valus is NaN, but the corresponding check is not enabled
        setmarginal!(x, NormalMeanPrecision(NaN, NaN))
        @test events[2] |> isnan

        # Third value is Inf, should trigger the check
        setmarginal!(x, NormalMeanPrecision(0.0, 0.0))
        @test events[3] isa String && occursin("The result is `Inf`", events[3])

        # Normally stream should unsubscribe after first error 
        setmarginal!(x, NormalMeanPrecision(0.0, 0.0))
        @test length(events) === 3
    end

    @testset "`BetheFreeEnergyCheckNaNs` for variable bound energy" begin

        # Dummy graph [U] - (x) - [U]
        model = FactorGraphModel()
        x     = randomvar(model, :x)
        make_node(Uninformative, FactorNodeCreationOptions(), x)
        make_node(Uninformative, FactorNodeCreationOptions(), x)

        activate!(model)

        objective = BetheFreeEnergy(IncludeAll(), BetheFreeEnergyCheckNaNs())

        vbenergy = score(InfCountingReal{Float64}, objective, VariableBoundEntropy(), x, AsapScheduler())

        events = []

        subscription = subscribe!(
            vbenergy |> safe(),
            lambda(
                on_next     = (data) -> push!(events, float(data)),
                on_error    = (err) -> push!(events, err),
                on_complete = () -> push!(events, "completed")
            )
        )

        # First value is ok
        setmarginal!(x, NormalMeanVariance(0.0, 1.0))
        @test events[1] ≈ 1.4189385332046727

        # Second valus is Inf, but the corresponding check is not enabled
        setmarginal!(x, NormalMeanPrecision(0.0, 0.0))
        @test events[2] |> isinf

        # Third value is NaN, should trigger the check
        setmarginal!(x, NormalMeanPrecision(NaN, NaN))
        @test events[3] isa String && occursin("The result is `NaN`", events[3])

        # Normally stream should unsubscribe after first error 
        setmarginal!(x, NormalMeanPrecision(0.0, 0.0))
        @test length(events) === 3
    end

    @testset "No checks for variable bound energy" begin

        # Dummy graph [U] - (x) - [U]
        model = FactorGraphModel()
        x     = randomvar(model, :x)
        make_node(Uninformative, FactorNodeCreationOptions(), x)
        make_node(Uninformative, FactorNodeCreationOptions(), x)

        activate!(model)

        objective = BetheFreeEnergy(IncludeAll(), nothing)

        vbenergy = score(InfCountingReal{Float64}, objective, VariableBoundEntropy(), x, AsapScheduler())

        events = []

        subscription = subscribe!(
            vbenergy |> safe(),
            lambda(
                on_next     = (data) -> push!(events, float(data)),
                on_error    = (err) -> push!(events, err),
                on_complete = () -> push!(events, "completed")
            )
        )

        # First value is ok
        setmarginal!(x, NormalMeanVariance(0.0, 1.0))
        @test events[1] ≈ 1.4189385332046727

        # Second valus is Inf, but the corresponding check is not enabled
        setmarginal!(x, NormalMeanPrecision(0.0, 0.0))
        @test events[2] |> isinf

        # Third value is NaN, but the corresponding check is not enabled
        setmarginal!(x, NormalMeanPrecision(NaN, NaN))
        @test events[3] |> isnan

        # Normally stream should not unsubscribe if there are no errors
        setmarginal!(x, NormalMeanPrecision(0.0, 0.0))
        @test length(events) === 4
    end

    @testset "`BetheFreeEnergyCheckInfs` for node bound energy" begin

        # Dummy graph [U] - (x) - [U]
        model = FactorGraphModel()
        x     = randomvar(model, :x)
        n1    = make_node(Uninformative, FactorNodeCreationOptions(), x)
        n2    = make_node(Uninformative, FactorNodeCreationOptions(), x)

        activate!(model)

        objective = BetheFreeEnergy(IncludeAll(), BetheFreeEnergyCheckInfs())

        nbenergy = score(InfCountingReal{Float64}, objective, FactorBoundFreeEnergy(), n1, AsapScheduler())

        events = []

        subscription = subscribe!(
            nbenergy |> safe(),
            lambda(
                on_next     = (data) -> push!(events, float(data)),
                on_error    = (err) -> push!(events, err),
                on_complete = () -> push!(events, "completed")
            )
        )

        # First value is ok, `entropy(d) - entropy(d)` === 0
        setmarginal!(x, NormalMeanVariance(0.0, 1.0))
        @test events[1] |> iszero

        # Second value is NaN, as `Inf - Inf` === NaN, but the corresponding check is disabled
        setmarginal!(x, NormalMeanVariance(0.0, 0.0))
        @test events[2] |> isnan

        struct DummyDistribution <: ContinuousUnivariateDistribution end

        # A hacky way to make `entropy(d) - entropy(d)` return Inf, dont use at home, for testing is fine
        count = 0
        Distributions.entropy(::DummyDistribution) = begin
            count += 1
            return count === 1 ? 0 : Inf
        end

        # Third value is Inf, should trigger the check
        setmarginal!(x, DummyDistribution())
        @test events[3] isa String && occursin("The result is `Inf`.", events[3])

        # Normally stream should unsubscribe after first error 
        setmarginal!(x, NormalMeanPrecision(0.0, 0.0))
        @test length(events) === 3
    end

    @testset "`BetheFreeEnergyCheckInfs` for node bound energy" begin

        # Dummy graph [U] - (x) - [U]
        model = FactorGraphModel()
        x     = randomvar(model, :x)
        n1    = make_node(Uninformative, FactorNodeCreationOptions(), x)
        n2    = make_node(Uninformative, FactorNodeCreationOptions(), x)

        activate!(model)

        objective = BetheFreeEnergy(IncludeAll(), BetheFreeEnergyCheckNaNs())

        nbenergy = score(InfCountingReal{Float64}, objective, FactorBoundFreeEnergy(), n1, AsapScheduler())

        events = []

        subscription = subscribe!(
            nbenergy |> safe(),
            lambda(
                on_next     = (data) -> push!(events, float(data)),
                on_error    = (err) -> push!(events, err),
                on_complete = () -> push!(events, "completed")
            )
        )

        # First value is ok, `entropy(d) - entropy(d)` === 0
        setmarginal!(x, NormalMeanVariance(0.0, 1.0))
        @test events[1] |> iszero

        struct DummyDistribution <: ContinuousUnivariateDistribution end

        # A hacky way to make `entropy(d) - entropy(d)` return Inf, dont use at home, for testing is fine
        count = 0
        Distributions.entropy(::DummyDistribution) = begin
            count += 1
            return count === 1 ? 0 : Inf
        end

        # Second value is Inf, but the corresponding check is disabled
        setmarginal!(x, DummyDistribution())
        @test events[2] |> isinf

        # Third value is NaN, as `Inf - Inf` === NaN, should trigger the check
        setmarginal!(x, NormalMeanVariance(0.0, 0.0))
        @test events[3] isa String && occursin("The result is `NaN`.", events[3])

        # Normally stream should unsubscribe after first error 
        setmarginal!(x, NormalMeanPrecision(0.0, 0.0))
        @test length(events) === 3
    end

    @testset "`BetheFreeEnergyCheckInfs` for node bound energy" begin

        # Dummy graph [U] - (x) - [U]
        model = FactorGraphModel()
        x     = randomvar(model, :x)
        n1    = make_node(Uninformative, FactorNodeCreationOptions(), x)
        n2    = make_node(Uninformative, FactorNodeCreationOptions(), x)

        activate!(model)

        objective = BetheFreeEnergy(IncludeAll(), nothing)

        nbenergy = score(InfCountingReal{Float64}, objective, FactorBoundFreeEnergy(), n1, AsapScheduler())

        events = []

        subscription = subscribe!(
            nbenergy |> safe(),
            lambda(
                on_next     = (data) -> push!(events, float(data)),
                on_error    = (err) -> push!(events, err),
                on_complete = () -> push!(events, "completed")
            )
        )

        # First value is ok, `entropy(d) - entropy(d)` === 0
        setmarginal!(x, NormalMeanVariance(0.0, 1.0))
        @test events[1] |> iszero

        struct DummyDistribution <: ContinuousUnivariateDistribution end

        # A hacky way to make `entropy(d) - entropy(d)` return Inf, dont use at home, for testing is fine
        count = 0
        Distributions.entropy(::DummyDistribution) = begin
            count += 1
            return count === 1 ? 0 : Inf
        end

        # Second value is Inf, but the corresponding check is disabled
        setmarginal!(x, DummyDistribution())
        @test events[2] |> isinf

        # Third value is NaN, as `Inf - Inf` === NaN, but the corresponding check is disabled
        setmarginal!(x, NormalMeanVariance(0.0, 0.0))
        @test events[3] |> isnan

        # Normally stream should not unsubscribe if there are no errors
        setmarginal!(x, NormalMeanPrecision(0.0, 0.0))
        @test length(events) === 4
    end
end

end
