using BenchmarkTools, LinearAlgebra, Statistics, Printf

benchmark_integer(name, default) = parse(Int, get(ENV, name, string(default)))
benchmark_workers() = unique([
    0,
    1,
    min(2, Threads.nthreads()),
    min(4, Threads.nthreads()),
    Threads.nthreads(),
])

# Do not retain the first inference graph for the lifetime of a Trial:
# BenchmarkTools.run internally retains the first expression's return value.
function benchmark_discard_result(run_inference, workers)
    run_inference(workers)
    return nothing
end

"""
    benchmark_multicore(run_inference, check_result; samples=5, rounds=2)

Benchmark a fresh inference graph per evaluation. `check_result(workers, result)`
runs outside the timed expression before measurement and after every trial.
It must not retain the graph; retain only posterior values if needed.

All configurations use evals=1, the same sample count and GC policy. BLAS is
the same by default; RXINFER_BENCHMARK_RUNNER_BLAS_THREADS can explicitly select
a different BLAS setting for the wave runner. Changes happen outside timing
and are logged per configuration. Round order alternates forward/reverse. No samples are
discarded. Timings include graph construction and GC during inference.
"""
function benchmark_multicore(
    run_inference,
    check_result;
    samples = benchmark_integer("RXINFER_BENCHMARK_SAMPLES", 5),
    rounds = benchmark_integer("RXINFER_BENCHMARK_ROUNDS", 2),
    workers = benchmark_workers(),
)
    samples > 0 || throw(ArgumentError("samples must be positive"))
    rounds > 0 || throw(ArgumentError("rounds must be positive"))
    first(workers) == 0 || throw(ArgumentError("standard runner must be first"))
    original_blas = BLAS.get_num_threads()
    selected_blas = benchmark_integer(
        "RXINFER_BENCHMARK_BLAS_THREADS", original_blas
    )
    selected_blas > 0 || throw(ArgumentError("BLAS threads must be positive"))
    runner_blas = benchmark_integer(
        "RXINFER_BENCHMARK_RUNNER_BLAS_THREADS", selected_blas
    )
    runner_blas > 0 ||
        throw(ArgumentError("runner BLAS threads must be positive"))
    blas_for(w) = iszero(w) ? selected_blas : runner_blas
    BLAS.set_num_threads(selected_blas)
    try
        println(
            "Julia=",
            VERSION,
            " BenchmarkTools=",
            pkgversion(BenchmarkTools),
            " CPU=",
            Sys.CPU_NAME,
            " Julia_threads=",
            Threads.nthreads(),
            " BLAS_threads=",
            BLAS.get_num_threads(),
            " runner_BLAS_threads=",
            runner_blas,
            " original_BLAS_threads=",
            original_blas,
        )
        println("active_project=", Base.active_project())
        if isdefined(@__MODULE__, :RxInfer)
            println("RxInfer_source=", pathof(RxInfer))
            println("ReactiveMP_source=", pathof(RxInfer.ReactiveMP))
        end
        runtime_options = Base.JLOptions()
        if hasproperty(runtime_options, :heap_size_hint)
            println("heap_size_hint_bytes=", runtime_options.heap_size_hint)
        end
        println(
            "evals=1 samples_per_round=",
            samples,
            " rounds=",
            rounds,
            " gctrial=true gcsample=true; timings include construction and inference",
        )
        benchmarks = Dict{Int, BenchmarkTools.Benchmark}()
        trials = BenchmarkGroup([
            "standard_blas=$selected_blas",
            "runner_blas=$runner_blas",
            "julia_threads=$(Threads.nthreads())",
        ])
        for w in workers
            BLAS.set_num_threads(blas_for(w))
            check_result(w, run_inference(w))
            benchmarks[w] = @benchmarkable benchmark_discard_result(
                $run_inference, $w
            ) evals=1
            trials[string(w)] = BenchmarkGroup()
            println(
                "WARM_AND_CHECK workers=",
                w,
                " BLAS_threads=",
                BLAS.get_num_threads(),
            )
            flush(stdout)
        end
        for round in 1:rounds
            order = isodd(round) ? workers : reverse(workers)
            for w in order
                BLAS.set_num_threads(blas_for(w))
                # A high time ceiling prevents the default five-second budget
                # silently giving slower configurations fewer samples.
                trial = run(
                    benchmarks[w];
                    samples,
                    seconds = 3600.0,
                    evals = 1,
                    gctrial = true,
                    gcsample = true,
                )
                length(trial) == samples || error(
                    "Benchmark time ceiling reached before collecting all samples",
                )
                trials[string(w)][string(round)] = trial
                check_result(w, run_inference(w))
                println(
                    "TRIAL round=",
                    round,
                    " workers=",
                    w,
                    " BLAS_threads=",
                    BLAS.get_num_threads(),
                )
                show(stdout, MIME("text/plain"), trial)
                println()
                flush(stdout)
            end
        end
        baseline = reduce(
            vcat, [trials["0"][string(r)].times for r in 1:rounds]
        )
        for w in workers
            parts = [trials[string(w)][string(r)] for r in 1:rounds]
            times = reduce(vcat, [part.times for part in parts]) ./ 1e9
            gctimes = reduce(vcat, [part.gctimes for part in parts]) ./ 1e9
            q25, q75 = quantile(times, [0.25, 0.75])
            @printf(
                "SUMMARY workers=%d BLAS_threads=%d samples=%d median_s=%.6f min_s=%.6f q25_s=%.6f q75_s=%.6f speedup=%.3f gc_median_s=%.6f bytes=%d allocs=%d\n",
                w,
                blas_for(w),
                length(times),
                median(times),
                minimum(times),
                q25,
                q75,
                median(baseline) / 1e9 / median(times),
                median(gctimes),
                minimum(memory.(parts)),
                minimum(allocs.(parts))
            )
            println("  samples_s=", times)
            println(
                "  round_medians_s=",
                [median(part).time / 1e9 for part in parts],
            )
        end
        println("peak_RSS_GB=", Sys.maxrss() / 1e9)
        if haskey(ENV, "RXINFER_BENCHMARK_OUTPUT")
            path = ENV["RXINFER_BENCHMARK_OUTPUT"]
            ispath(path) &&
                error("Refusing to overwrite benchmark output: $path")
            BenchmarkTools.save(path, trials)
            println("Saved raw BenchmarkTools trials to ", path)
        end
        return trials
    finally
        BLAS.set_num_threads(original_blas)
    end
end
