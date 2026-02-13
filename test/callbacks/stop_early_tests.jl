@testitem "Tests for `StopEarlyIterationStrategy`" begin
    using RxInfer, Rocket

    strategy = StopEarlyIterationStrategy(1e-3)
    strategy_with_atol = StopEarlyIterationStrategy(1e-5, 1e-3)

    @test strategy.atol === 0.0
    @test strategy.rtol === 1e-3
    @test strategy.start_fe_value === Inf
    @test isempty(strategy.fe_values)

    @test strategy_with_atol.atol === 1e-5
    @test strategy_with_atol.rtol === 1e-3
    @test strategy_with_atol.start_fe_value === Inf
    @test isempty(strategy_with_atol.fe_values)

    include("mock_model.jl")
    # Create a subject you can push values into
    fe_subject = Rocket.ReplaySubject(Float64, 1)

    mock = MockModel(fe_subject)

    # Push a free energy value
    next!(fe_subject, 100.0)
    @test strategy(mock, 1) == false

    # Free energy value is close to the previous one, should stop
    next!(fe_subject, 100.0)
    @test strategy(mock, 2) == true

    # Free energy value is not close to the previous one, should not stop
    next!(fe_subject, 110.0)
    @test strategy(mock, 3) == false
end
