@testitem "Session can be created" begin
    using TOML

    session = RxInfer.create_session()

    @test hasproperty(session, :id)
    @test hasproperty(session, :created_at)
    @test hasproperty(session, :invokes)
    @test hasproperty(session, :environment)

    # Empty session has no invokes
    @test length(session.invokes) == 0

    # Version info should contain all required fields
    @test haskey(session.environment, :julia_version)
    @test haskey(session.environment, :rxinfer_version)
    @test haskey(session.environment, :os)
    @test haskey(session.environment, :machine)
    @test haskey(session.environment, :cpu_threads)
    @test haskey(session.environment, :word_size)

    # Version info should have correct types and values
    @test session.environment[:julia_version] == string(VERSION)
    @test session.environment[:os] == string(Sys.KERNEL)
    @test session.environment[:machine] == string(Sys.MACHINE)
    @test session.environment[:cpu_threads] == Sys.CPU_THREADS
    @test session.environment[:word_size] == Sys.WORD_SIZE

    rxinfer_version = 
        VersionNumber(TOML.parsefile(joinpath(pkgdir(RxInfer), "Project.toml"))["version"])
    @test session.environment[:rxinfer_version] == string(rxinfer_version)
end

@testitem "RxInfer should have a default session" begin
    default_session = RxInfer.default_session()

    @test hasproperty(default_session, :id)
    @test hasproperty(default_session, :created_at)
    @test hasproperty(default_session, :environment)
    @test hasproperty(default_session, :invokes)

    # Check second invokation doesn't change the return value
    @test default_session === RxInfer.default_session()
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

@testitem "log_data_entry" begin
    import RxInfer: log_data_entry

    @testset "Scalar values" begin
        let entry = log_data_entry(:y => 1)
            @test entry.name === :y
            @test entry.type === Int
            @test entry.size === ()
            @test entry.elsize === ()
        end

        let entry = log_data_entry(:x => 3.14)
            @test entry.name === :x
            @test entry.type === Float64
            @test entry.size === ()
            @test entry.elsize === ()
        end
    end

    @testset "Vectors" begin
        let entry = log_data_entry(:x => [1])
            @test entry.name === :x
            @test entry.type === Vector{Int}
            @test entry.size === (1,)
            @test entry.elsize === ()
        end

        let entry = log_data_entry(:x => [1.0, 2.0, 3.0])
            @test entry.name === :x
            @test entry.type === Vector{Float64}
            @test entry.size === (3,)
            @test entry.elsize === ()
        end

        let entry = log_data_entry(:x => [[1, 2], [3, 4]])
            @test entry.name === :x
            @test entry.type === Vector{Vector{Int}}
            @test entry.size === (2,)
            @test entry.elsize === (2,)
        end
    end

    @testset "Matrices" begin
        let entry = log_data_entry(:x => ones(2, 3))
            @test entry.name === :x
            @test entry.type === Matrix{Float64}
            @test entry.size === (2, 3)
            @test entry.elsize === ()
        end

        let entry = log_data_entry(:x => reshape([1, 2, 3, 4], 2, 2))
            @test entry.name === :x
            @test entry.type === Matrix{Int}
            @test entry.size === (2, 2)
            @test entry.elsize === ()
        end
    end

    @testset "Matrix of vectors" begin
        let data = Matrix{Vector{Float64}}(undef, 2, 2)
            data[1, 1] = [1.0, 2.0]
            data[1, 2] = [3.0, 4.0]
            data[2, 1] = [5.0, 6.0]
            data[2, 2] = [7.0, 8.0]
            entry = log_data_entry(:x => data)
            @test entry.name === :x
            @test entry.type === Matrix{Vector{Float64}}
            @test entry.size === (2, 2)
            @test entry.elsize === (2,)
        end
    end

    struct StrangeDataEntry end

    @testset let entry = log_data_entry(StrangeDataEntry)
        @test entry.name === :unknown
        @test entry.type === :unknown
        @test entry.size === :unknown
        @test entry.elsize === :unknown
    end
end

@testitem "log_data_entries" begin
    import RxInfer: log_data_entry, log_data_entries

    @testset "Named tuple entries" begin
        data = (y = 1, x = [2.0, 3.0], z = [[1.0, 2.0], [3.0]])
        entries = log_data_entries(data)

        @test length(entries) === 3

        # Check y entry
        y_entry = entries[1]
        @test y_entry.name === :y
        @test y_entry.type === Int
        @test y_entry.size === ()
        @test y_entry.elsize === ()

        # Check x entry
        x_entry = entries[2]
        @test x_entry.name === :x
        @test x_entry.type === Vector{Float64}
        @test x_entry.size === (2,)
        @test x_entry.elsize === ()

        # Check z entry
        z_entry = entries[3]
        @test z_entry.name === :z
        @test z_entry.type === Vector{Vector{Float64}}
        @test z_entry.size === (2,)
        @test z_entry.elsize === (2,)
    end

    @testset "Dictionary entries" begin
        data = Dict(:y => 1, :x => [2.0, 3.0], :z => [[1.0, 2.0], [3.0]])
        entries = log_data_entries(data)

        @test length(entries) === 3
        @test Set(entry.name for entry in entries) == Set([:x, :y, :z])

        # Find and check y entry
        y_entry = findfirst(e -> e.name === :y, entries)
        @test !isnothing(y_entry)
        y_entry = entries[y_entry]
        @test y_entry.type === Int
        @test y_entry.size === ()
        @test y_entry.elsize === ()

        # Find and check x entry
        x_entry = findfirst(e -> e.name === :x, entries)
        @test !isnothing(x_entry)
        x_entry = entries[x_entry]
        @test x_entry.type === Vector{Float64}
        @test x_entry.size === (2,)
        @test x_entry.elsize === ()

        # Find and check z entry
        z_entry = findfirst(e -> e.name === :z, entries)
        @test !isnothing(z_entry)
        z_entry = entries[z_entry]
        @test z_entry.type === Vector{Vector{Float64}}
        @test z_entry.size === (2,)
        @test z_entry.elsize === (2,)
    end

    struct UnknownStruct end
    @test log_data_entries(UnknownStruct()) == :unknown # be safe on something we don't know how to parse

    @testset "data with UnknownStructs as elements" begin 
        data = (y = UnknownStruct(), x = UnknownStruct())

        entries = log_data_entries(data)

        @test length(entries) === 2

        # Check y entry
        y_entry = entries[1]
        @test y_entry.name === :y
        @test y_entry.type === UnknownStruct
        @test y_entry.size === :unknown
        @test y_entry.elsize === :unknown

        # Check x entry
        x_entry = entries[2]
        @test x_entry.name === :x
        @test x_entry.type === UnknownStruct
        @test x_entry.size === :unknown
        @test x_entry.elsize === :unknown
    end
end

@testitem "Session Logging basic execution" begin

    # Create a simple model for testing
    @model function simple_model(y)
        x ~ Normal(mean = 0.0, var = 1.0)
        y ~ Normal(mean = x, var = 1.0)
    end

    # Create test data
    test_data = (y = 1.0,)

    # Run inference inside session `session`
    result = infer(model = simple_model(), data = test_data)

    session = RxInfer.default_session()

    # Basic checks, other tests may have produced more invokes here
    @test length(session.invokes) >= 1

    # Check the latest invoke
    latest_invoke = session.invokes[end]
    @test !isempty(latest_invoke.id)
    @test latest_invoke.status == :success
    @test latest_invoke.execution_end > latest_invoke.execution_start
    @test hasproperty(latest_invoke.context, :model)
    @test hasproperty(latest_invoke.context, :data)
    @test !isnothing(latest_invoke.context.data)
    @test latest_invoke.context.model == """
    function simple_model(y)
        x ~ Normal(0.0, 1.0)
        y ~ Normal(x, 1.0)
    end"""
    @test length(latest_invoke.context.data) === 1

    # Check saved properties of the passed data `y`
    saved_data_properties = latest_invoke.context.data[end]
    @test saved_data_properties.name === :y
    @test saved_data_properties.type === Int

    custom_session = RxInfer.create_session()
    result = infer(model = simple_model(), data = test_data, session = session)

    @test length(custom_session.invokes) === 1
    @test latest_invoke.id != custom_session.invokes[1].id
    @test latest_invoke.context == custom_session.invokes[1].context
end