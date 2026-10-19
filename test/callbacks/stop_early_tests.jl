@testitem "Tests for `StopEarlyIterationStrategy`" begin
    using RxInfer, Rocket, UUIDs

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
    event1 = AfterIterationEvent(mock, 1, uuid4())
    strategy(event1)
    @test event1.stop_iteration === false

    # Free energy value is close to the previous one, should stop
    next!(fe_subject, 100.0)
    event2 = AfterIterationEvent(mock, 2, uuid4())
    strategy(event2)
    @test event2.stop_iteration === true

    # Free energy value is not close to the previous one, should not stop
    next!(fe_subject, 110.0)
    event3 = AfterIterationEvent(mock, 3, uuid4())
    strategy(event3)
    @test event3.stop_iteration === false

    @test strategy.window == 2
    @test StopEarlyIterationStrategy(0, 0, Inf, Float64[]).window == 2
    @test StopEarlyIterationStrategy(1e-3; window = 4).window == 4
    @test_throws ArgumentError StopEarlyIterationStrategy(1e-3; window = 1)
    @test_throws ArgumentError StopEarlyIterationStrategy(-1.0)
    @test_throws ArgumentError StopEarlyIterationStrategy(Inf)
    @test_throws ArgumentError StopEarlyIterationStrategy(NaN, 0.0)

    function check_sequence(values; atol = 0.1, rtol = 0.0, window = 4)
        callback = StopEarlyIterationStrategy(atol, rtol; window)
        stopped = Bool[]
        for (i, value) in enumerate(values)
            next!(fe_subject, value)
            event = AfterIterationEvent(mock, i, uuid4())
            callback(event)
            push!(stopped, event.stop_iteration)
        end
        return callback, stopped
    end

    @test last(check_sequence([1.0, 1.0, 1.0, 1.0])) ==
        [false, false, false, true]
    # Adjacent differences pass, but the full window detects accumulated drift.
    @test !any(last(check_sequence([1.0, 1.06, 1.12, 1.18])))
    @test !any(last(check_sequence([1.0, 1.2, 1.0, 1.2])))
    @test last(
        check_sequence(
            [-100.0, -100.01, -100.02, -100.03]; atol = 0.0, rtol = 1e-3
        ),
    )[end]
    @test !any(last(check_sequence([Inf, Inf, Inf, Inf])))
    @test !any(last(check_sequence([NaN, NaN, NaN, NaN])))
    @test !any(last(check_sequence([Inf, Inf]; window = 2)))
    @test last(check_sequence([0.0, 0.0, 0.0, 0.0]; atol = 0.0))[end]
    callback, _ = check_sequence([1.0, 1.0, 1.0, 1.0])
    next!(fe_subject, 1.0)
    restarted = AfterIterationEvent(mock, 1, uuid4())
    callback(restarted)
    @test !restarted.stop_iteration
    @test callback.fe_values == [1.0]
    unavailable = StopEarlyIterationStrategy(1e-3)
    missing_event = AfterIterationEvent(MockModel(Subject(Float64)), 1, uuid4())
    @test_throws ErrorException unavailable(missing_event)
    @test isempty(unavailable.fe_values)
    @test !last(
        check_sequence(
            [100.0, 100.04, 100.08, 100.12]; atol = 0.05, rtol = 1e-3
        ),
    )[end]
end
