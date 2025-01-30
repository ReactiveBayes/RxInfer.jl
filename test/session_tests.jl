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

    if get(ENV, "CI", "false") === "true"
        # This test breaks precompilation in VSCode, thus disabled locally, executes only in CI
        rxinfer_version = VersionNumber(TOML.parsefile(joinpath(pkgdir(RxInfer), "Project.toml"))["version"])
        @test session.environment[:rxinfer_version] == string(rxinfer_version)
    end
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

@testitem "Log session should save the context" begin
    session = RxInfer.create_session()
    result = RxInfer.with_session(session) do invoke
        RxInfer.append_invoke_context(invoke) do ctx
            ctx[:a] = 1
            ctx[:b] = 2
        end
        return 3
    end
    @test length(session.invokes) === 1
    last_invoke = session.invokes[end]
    @test last_invoke.context[:a] === 1
    @test last_invoke.context[:b] === 2
    @test result === 3

    result = RxInfer.with_session(nothing) do invoke
        RxInfer.append_invoke_context(invoke) do ctx
            ctx[:a] = 1
            ctx[:b] = 2
        end
        return 4
    end
    @test result === 4
end

@testitem "Log session should save errors if any" begin
    session = RxInfer.create_session()
    @test_throws "I'm an error" RxInfer.with_session(session) do invoke
        error("I'm an error")
    end
    @test length(session.invokes) === 1
    last_invoke = session.invokes[end]
    @test last_invoke.context[:error] == "ErrorException(\"I'm an error\")"

    @test_throws "I'm an error" RxInfer.with_session(nothing) do invoke
        error("I'm an error")
    end
end