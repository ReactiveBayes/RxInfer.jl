# Accuracy-matched benchmark. Run alone, never alongside capacity probes/tests.
include("compiled_grid_capacity.jl")
using Rocket, BenchmarkTools, Statistics, Dates, JSON, SHA

function compiled_benchmark_model(problem)
    (; diagonal_entries, sources, destinations, couplings) = problem
    return compiled_edge_linear_system(; diagonal_entries, sources, destinations, couplings)
end

function compiled_benchmark_fixture(problem, workers)
    model, data = compiled_benchmark_model(problem), (b=problem.b,)
    config = convert(RxInfer.ReactiveMPInferenceOptions, (limit_stack_depth=100,))
    if workers == 0
        plugins = GraphPPL.PluginsCollection(GraphPPL.VariationalConstraintsPlugin(), GraphPPL.MetaPlugin(nothing),
            RxInfer.InitializationPlugin(linear_initialization), RxInfer.ReactiveMPInferencePlugin(config))
        generator = GraphPPL.with_backend(GraphPPL.with_plugins(model, plugins), RxInfer.ReactiveMPGraphPPLBackend(RxInfer.Static.static(false)))
        fmodel = RxInfer.create_model(generator | data)
        variables = GraphPPL.variables(getvardict(fmodel))
        actor = RxInfer.make_actor(variables[:x], KeepLast())
        subscription = Rocket.subscribe!(RxInfer.obtain_marginal(variables[:x]), actor)
        return (; model=fmodel, program=nothing, data=RxInfer.getvariable(variables[:b]),
            slots=nothing, actor, subscription, posterior=Ref{Any}(nothing))
    end
    builder, _ = RxInfer.compiled_model_builder(model, data, linear_initialization, nothing, nothing,
        config, CompiledRunner(; workers), nothing)
    RxInfer.compiled_lower_variables!(builder)
    slots = RxInfer.compiled_selected_slots(builder.model, (x=KeepLast(),), false)[:x]
    RxInfer.compiled_finalize!(builder, vec(slots))
    return (; model=builder.model, program=builder.program, data, slots,
        actor=nothing, subscription=Rocket.VoidTeardown(), posterior=Ref{Any}(nothing))
end
compiled_benchmark_close(fixture) = Rocket.unsubscribe!(fixture.subscription)

function compiled_benchmark_inference!(fixture, problem, iterations)
    if fixture.program === nothing
        for _ in 1:iterations
            RxInfer.new_observation_indexed!(fixture.data, problem.b)
        end
        fixture.posterior[] = RxInfer.inference_postprocess(RxInfer.UnpackMarginalPostprocess(), Rocket.getvalues(fixture.actor))
    else
        for _ in 1:iterations
            RxInfer.compiled_update_data!(fixture.model, fixture.program, fixture.data)
            ReactiveMP.compiled_sweep!(fixture.program)
            fixture.posterior[] = RxInfer.inference_postprocess(RxInfer.UnpackMarginalPostprocess(),
                RxInfer.compiled_snapshot(fixture.program, fixture.slots))
        end
    end
    return nothing
end

function compiled_benchmark_calibrate(problem, workers; max_sweeps=5000)
    fixture = compiled_benchmark_fixture(problem, workers)
    previous_means, previous_vars, scratch = [zeros(length(problem.b)) for _ in 1:3]
    stable = 0
    try
        for iteration in 1:max_sweeps
            compiled_benchmark_inference!(fixture, problem, 1)
            means, vars = mean.(fixture.posterior[]), var.(fixture.posterior[])
            settled = all(isapprox.(means, previous_means; rtol=1e-4, atol=1e-6)) &&
                all(isapprox.(vars, previous_vars; rtol=1e-4, atol=1e-6))
            stable = settled ? stable + 1 : 0
            residual = compiled_grid_residual!(scratch, problem, means)
            if stable >= 5 && residual <= 1e-6
                println("CALIBRATED workers=", workers, " sweeps=", iteration, " residual=", residual, " stable=", stable)
                flush(stdout)
                return (; iterations=iteration, means, vars, residual, stable)
            end
            copyto!(previous_means, means); copyto!(previous_vars, vars)
        end
        error("Calibration failed to converge in $max_sweeps sweeps")
    finally
        compiled_benchmark_close(fixture)
    end
end

function compiled_benchmark_batch(problem, workers, iterations)
    options = workers == 0 ? (limit_stack_depth=100,) : (runner=CompiledRunner(; workers),)
    return infer(model=compiled_benchmark_model(problem), data=(b=problem.b,), initialization=linear_initialization,
        options=options, iterations=iterations, returnvars=(x=KeepLast(),), session=nothing)
end
function compiled_benchmark_build_discard(problem, workers)
    fixture = compiled_benchmark_fixture(problem, workers)
    compiled_benchmark_close(fixture)
    return nothing
end
function compiled_benchmark_batch_discard(problem, workers, iterations)
    compiled_benchmark_batch(problem, workers, iterations)
    return nothing
end

function compiled_benchmark_check(problem, posterior, reference)
    means, vars = mean.(posterior), var.(posterior)
    @assert compiled_grid_residual!(zeros(length(means)), problem, means) <= 1e-6
    @assert isapprox(means, reference.means; rtol=1e-4, atol=1e-6)
    @assert isapprox(vars, reference.vars; rtol=1e-4, atol=1e-6)
    @assert all(>(0), vars)
end

function compiled_benchmark_main(side; measure=true)
    problem = compiled_grid_problem(side)
    configs = [(workers=0, blas=6), (workers=1, blas=1), (workers=Threads.nthreads(), blas=1), (workers=Threads.nthreads(), blas=6)]
    println("Julia=", VERSION, " BenchmarkTools=", pkgversion(BenchmarkTools), " CPU=", Sys.CPU_NAME,
        " threads=", Threads.nthreads(), " project=", Base.active_project(), " side=", side)
    println("Float64 leak=0.1 RHS=sin.(1:n) KeepLast; evals=1 samples=3 rounds=2 gctrial=true gcsample=true")
    calibrated = []
    for config in configs
        BLAS.set_num_threads(config.blas)
        calibration = compiled_benchmark_calibrate(problem, config.workers)
        push!(calibrated, calibration)
        result = compiled_benchmark_batch(problem, config.workers, calibration.iterations)
        compiled_benchmark_check(problem, result.posteriors[:x], first(calibrated))
        # Verify that the isolated inference fixture reproduces the public API.
        fixture = compiled_benchmark_fixture(problem, config.workers)
        try
            compiled_benchmark_inference!(fixture, problem, calibration.iterations)
            compiled_benchmark_check(problem, fixture.posterior[], first(calibrated))
        finally
            compiled_benchmark_close(fixture)
        end
    end
    measure || return nothing
    trials = BenchmarkGroup()
    # Julia removes mktempdir() at process exit by default. Keep the raw samples
    # available for independent inspection after a completed benchmark.
    output_root = mkpath(joinpath(@__DIR__, "compiled", "results"))
    output_path = joinpath(mktempdir(output_root; prefix="grid$(side)-", cleanup=false), "trials.json")
    # Pair the raw samples with enough context to interpret/reproduce the run.
    sources = Dict{String, String}()
    for package in (RxInfer, ReactiveMP, GraphPPL)
        for (directory, _, files) in walkdir(joinpath(pkgdir(package), "src")), file in sort(files)
            endswith(file, ".jl") || continue
            path = joinpath(directory, file)
            sources[string(nameof(package), "/", relpath(path, pkgdir(package)))] = bytes2hex(sha256(read(path)))
        end
    end
    metadata = (; timestamp_utc=string(now(UTC)), julia=string(VERSION),
        benchmarktools=string(pkgversion(BenchmarkTools)), cpu=Sys.CPU_NAME,
        threads=Threads.nthreads(), project=Base.active_project(), side,
        manifest_sha256=bytes2hex(sha256(read(joinpath(dirname(Base.active_project()), "Manifest.toml")))),
        harness_sha256=bytes2hex(sha256(read(@__FILE__))), sources,
        samples_per_round=3, rounds=2, evals=1, gctrial=true, gcsample=true,
        residual_tolerance=1e-6, stability_rtol=1e-4, stability_atol=1e-6, stable_sweeps_required=5,
        configurations=[(; config..., iterations=c.iterations, residual=c.residual, stable=c.stable)
            for (config, c) in zip(configs, calibrated)])
    open(joinpath(dirname(output_path), "metadata.json"), "w") do io
        JSON.print(io, metadata, 2)
    end
    println("RAW_TRIALS ", output_path)
    flush(stdout)
    for round in 1:2, i in (round == 1 ? eachindex(configs) : reverse(eachindex(configs)))
        config, iterations = configs[i], calibrated[i].iterations
        workers = config.workers
        BLAS.set_num_threads(config.blas)
        for phase in (:construction, :inference, :end_to_end)
            benchmark = if phase === :construction
                @benchmarkable compiled_benchmark_build_discard($problem, $workers) evals=1
            elseif phase === :inference
                @benchmarkable compiled_benchmark_inference!(fixture, $problem, $iterations) setup=(fixture=compiled_benchmark_fixture($problem, $workers)) teardown=(compiled_benchmark_close(fixture)) evals=1
            else
                @benchmarkable compiled_benchmark_batch_discard($problem, $workers, $iterations) evals=1
            end
            trial = run(benchmark; samples=3, seconds=3600, evals=1, gctrial=true, gcsample=true)
            @assert length(trial) == 3
            trials["workers=$workers/blas=$(config.blas)/round=$round/phase=$phase"] = trial
            BenchmarkTools.save(output_path, trials)
            estimate = median(trial)
            @printf("COMPILED_BT side=%d workers=%d blas=%d round=%d phase=%s sweeps=%d median_ms=%.6f bytes=%d allocations=%d gc_ms=%.6f\n",
                side, workers, config.blas, round, phase, iterations, estimate.time/1e6, estimate.memory, estimate.allocs, estimate.gctime/1e6)
            flush(stdout)
            result = compiled_benchmark_batch(problem, workers, iterations)
            compiled_benchmark_check(problem, result.posteriors[:x], first(calibrated))
        end
    end
    return nothing
end
abspath(PROGRAM_FILE) == (@__FILE__) && compiled_benchmark_main(isempty(ARGS) ? 64 : parse(Int, ARGS[1]); measure=!("--check-only" in ARGS))
