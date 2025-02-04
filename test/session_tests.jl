@testitem "Session can be created" begin
    using TOML, DataStructures

    session = RxInfer.create_session()

    @test hasproperty(session, :id)
    @test hasproperty(session, :created_at)
    @test hasproperty(session, :stats)
    @test hasproperty(session, :environment)

    # Empty session has no stats in the beginning
    @test length(session.stats) == 0

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

    if get(ENV, "CI", "false") == "true"
        # This test breaks precompilation in VSCode, thus disabled locally, executes only in CI
        rxinfer_version = VersionNumber(TOML.parsefile(joinpath(pkgdir(RxInfer), "Project.toml"))["version"])
        @test session.environment[:rxinfer_version] == string(rxinfer_version)
    end
end

@testitem "SessionStats should have capacity limits" begin
    using DataStructures

    # Test default capacity
    default_capacity = RxInfer.DEFAULT_SESSION_STATS_CAPACITY
    session = RxInfer.create_session()
    stats = RxInfer.get_session_stats(session, :for_testing)

    @test capacity(stats.invokes) == default_capacity

    # Test circular behavior
    for i in 1:(default_capacity + 1)
        invoke = RxInfer.create_invoke()
        RxInfer.update_session!(session, :for_testing, invoke)
    end

    # Should only keep last `default_capacity`
    @test length(stats.invokes) == default_capacity
end

@testitem "RxInfer should have a default session" begin
    default_session = RxInfer.default_session()

    @test hasproperty(default_session, :id)
    @test hasproperty(default_session, :created_at)
    @test hasproperty(default_session, :environment)
    @test hasproperty(default_session, :stats)

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

@testitem "Log session should save the context" begin
    session = RxInfer.create_session()
    result = RxInfer.with_session(session, :testing_session) do invoke
        RxInfer.append_invoke_context(invoke) do ctx
            ctx[:a] = 1
            ctx[:b] = 2
        end
        return 3
    end
    stats = RxInfer.get_session_stats(session, :testing_session)
    @test length(stats.invokes) === 1
    last_invoke = stats.invokes[end]
    @test last_invoke.context[:a] === 1
    @test last_invoke.context[:b] === 2
    @test result === 3

    result = RxInfer.with_session(nothing, :testing_session) do invoke
        RxInfer.append_invoke_context(invoke) do ctx
            ctx[:a] = 1
            ctx[:b] = 2
        end
        return 4
    end
    @test result === 4
    @test length(stats.invokes) === 1
end

@testitem "Log session should save errors if any" begin
    session = RxInfer.create_session()
    @test_throws "I'm an error" RxInfer.with_session(session, :error_session) do invoke
        error("I'm an error")
    end
    stats = RxInfer.get_session_stats(session, :error_session)
    @test length(stats.invokes) === 1
    last_invoke = stats.invokes[end]
    @test last_invoke.context[:error] == "ErrorException(\"I'm an error\")"

    @test_throws "I'm an error" RxInfer.with_session(nothing, :error_session) do invoke
        error("I'm an error")
    end
    @test length(stats.invokes) === 1
end

@testitem "Real-time session statistics" begin
    using Dates

    session = RxInfer.create_session()

    # Test initial empty state
    @test isempty(session.stats)
    empty_stats = RxInfer.get_session_stats(session, :test)
    @test empty_stats.total_invokes == 0
    @test empty_stats.success_count == 0
    @test empty_stats.failed_count == 0
    @test empty_stats.success_rate == 0.0
    @test empty_stats.total_duration_ms == 0.0
    @test empty_stats.min_duration_ms == Inf
    @test empty_stats.max_duration_ms == -Inf
    @test isempty(empty_stats.context_keys)

    # Create test invokes with controlled durations
    start_time = now()

    # First invoke: 100ms duration
    invoke1 = RxInfer.create_invoke()
    invoke1.status = :success
    invoke1.context[:key1] = "value1"
    invoke1.execution_start = start_time
    invoke1.execution_end = start_time + Millisecond(100)

    # Test after first successful invoke
    RxInfer.update_session!(session, :session_stats_test, invoke1)
    stats1 = RxInfer.get_session_stats(session, :session_stats_test)
    @test stats1.total_invokes == 1
    @test stats1.success_count == 1
    @test stats1.failed_count == 0
    @test stats1.success_rate == 1.0
    @test stats1.total_duration_ms == 100.0
    @test stats1.min_duration_ms == 100.0
    @test stats1.max_duration_ms == 100.0
    @test stats1.context_keys == Set([:key1])

    # Second invoke: 200ms duration
    invoke2 = RxInfer.create_invoke()
    invoke2.status = :error
    invoke2.context[:key2] = "value2"
    invoke2.context[:error] = "test error"
    invoke2.execution_start = start_time + Millisecond(200)
    invoke2.execution_end = start_time + Millisecond(400)  # 200ms duration

    # Test after error invoke
    RxInfer.update_session!(session, :session_stats_test, invoke2)
    stats2 = RxInfer.get_session_stats(session, :session_stats_test)
    @test stats2.total_invokes == 2
    @test stats2.success_count == 1
    @test stats2.failed_count == 1
    @test stats2.success_rate == 0.5
    @test stats2.total_duration_ms == 300.0  # 100ms + 200ms
    @test stats2.min_duration_ms == 100.0
    @test stats2.max_duration_ms == 200.0
    @test stats2.context_keys == Set([:key1, :key2, :error])

    # Third invoke: 50ms duration (shortest)
    invoke3 = RxInfer.create_invoke()
    invoke3.status = :success
    invoke3.context[:key3] = "value3"
    invoke3.execution_start = start_time + Millisecond(500)
    invoke3.execution_end = start_time + Millisecond(550)  # 50ms duration

    # Test after quick successful invoke
    RxInfer.update_session!(session, :session_stats_test, invoke3)
    stats3 = RxInfer.get_session_stats(session, :session_stats_test)
    @test stats3.total_invokes == 3
    @test stats3.success_count == 2
    @test stats3.failed_count == 1
    @test stats3.success_rate ≈ 2 / 3
    @test stats3.total_duration_ms == 350.0  # 100ms + 200ms + 50ms
    @test stats3.min_duration_ms == 50.0
    @test stats3.max_duration_ms == 200.0
    @test stats3.context_keys == Set([:key1, :key2, :key3, :error])

    # Test multiple labels with 150ms duration
    other_invoke = RxInfer.create_invoke()
    other_invoke.status = :success
    other_invoke.context[:other_key] = "other_value"
    other_invoke.execution_start = start_time + Millisecond(600)
    other_invoke.execution_end = start_time + Millisecond(750)  # 150ms duration

    RxInfer.update_session!(session, :other_session_stats_test, other_invoke)
    other_stats = RxInfer.get_session_stats(session, :other_session_stats_test)
    @test other_stats.total_invokes == 1
    @test other_stats.success_count == 1
    @test other_stats.failed_count == 0
    @test other_stats.success_rate == 1.0
    @test other_stats.total_duration_ms == 150.0
    @test other_stats.min_duration_ms == 150.0
    @test other_stats.max_duration_ms == 150.0
    @test other_stats.context_keys == Set([:other_key])

    # Verify original stats unchanged
    final_test_stats = RxInfer.get_session_stats(session, :session_stats_test)
    @test final_test_stats === stats3  # Should be exactly the same object
end

@testitem "Show methods should produce expected output" begin
    using Dates

    # Test SessionInvoke show
    invoke = RxInfer.create_invoke()
    invoke.status = :success
    invoke.execution_end = invoke.execution_start + Millisecond(123)

    output = sprint(show, invoke)
    @test occursin("SessionInvoke(id=$(invoke.id), status=success, duration=123.0ms", output)

    # Test SessionStats show
    stats = RxInfer.SessionStats(:test)
    RxInfer.update_stats!(stats, invoke)

    output = sprint(show, stats)
    @test occursin("SessionStats(id=$(stats.id), label=:test, total=1, success_rate=100.0%, invokes=1/$(RxInfer.DEFAULT_SESSION_STATS_CAPACITY))", output)

    # Test Session show
    session = RxInfer.create_session()
    RxInfer.update_session!(session, :test, invoke)
    RxInfer.update_session!(session, :other, invoke)

    output = sprint(show, session)
    @test occursin("Session(id=$(session.id), labels=", output)
    @test occursin("test", output)
    @test occursin("other", output)
end
