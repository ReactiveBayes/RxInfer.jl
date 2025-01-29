
@testitem "Session can be created" begin 
    session = RxInfer.create_session()

    @test hasproperty(session, :id)
    @test hasproperty(session, :created_at)
    @test hasproperty(session, :invokes)

    # Empty session has no invokes
    @test length(session.invokes) == 0
end

@testitem "RxInfer should have a default session" begin 
    default_session = RxInfer.default_session()

    @test hasproperty(default_session, :id)
    @test hasproperty(default_session, :created_at)
    @test hasproperty(default_session, :invokes)
end

@testitem "It should be possible to change the default session" begin 
    original_default_session = RxInfer.default_session()
    new_session = RxInfer.create_session()

    RxInfer.set_default_session!(new_session)

    new_default_session = RxInfer.default_session()

    @test new_default_session != original_default_session
    @test new_default_session.id != original_default_session.id
    @test new_default_session.created_at > original_default_session.created_at
end

@testitem "Session Logging basic execution" begin

    # Create a simple model for testing
    @model function simple_model()
        x ~ Normal(0.0, 1.0)
        y ~ Normal(x, 1.0)
        return y
    end

    # Create test data
    test_data = (y = 1.0,)

    session = RxInfer.create_session()

    # Run inference inside session `session`
    result = infer(
        model = simple_model(), 
        data = test_data,
        session = session
    )

    # Basic checks
    @test length(session_after.invokes) == 1

    # Check the latest invoke
    latest_invoke = session_after.invokes[end]
    @test !isempty(latest_invoke.id)
    @test latest_invoke.status == :success
    @test latest_invoke.execution_time > 0.0
    @test hasproperty(latest_invoke.context, :model)
    @test hasproperty(latest_invoke.context, :data)
    @test length(latest_invoke.context.data) === 1

    # Check saved properties of the passed data `y`
    saved_data_properties = latest_invoke.context.data[end]
    @test saved_data_properties.name === :y
    @test saved_data_properties.type === Int
    @test saved_data_properties.length === 1
end