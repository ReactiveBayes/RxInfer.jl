@testitem "Tests for `StopEarlyIterationStrategy`" begin
    using RxInfer, Rocket

    strategy = StopEarlyIterationStrategy(1e-3)

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
