# Capacity/correctness probe, NOT a timing benchmark. Run in a fresh process:
# julia --project=<local-dev-env> --threads=6 --heap-size-hint=5G benchmarks/compiled_grid_capacity.jl 500 6
ENV["LOG_USING_RXINFER"] = "false"
using RxInfer, LinearAlgebra, Printf
include("multicore_linear_system.jl")

@model function compiled_edge_linear_system(b, diagonal_entries, sources, destinations, couplings)
    for i in eachindex(b)
        x[i] ~ Normal(mean = b[i] / diagonal_entries[i], precision = diagonal_entries[i])
    end
    for k in eachindex(sources)
        x[destinations[k]] ~ GaussianCoupling(x[sources[k]], couplings[k])
    end
end

function compiled_grid_problem(side; leak = 0.1)
    side >= 2 || throw(ArgumentError("side must be at least 2"))
    n = side^2
    diagonal_entries = Vector{Float64}(undef, n)
    sources, destinations, couplings = Int[], Int[], Float64[]
    for row in 1:side, col in 1:side
        i = (row - 1) * side + col
        diagonal_entries[i] = leak + (row > 1) + (row < side) + (col > 1) + (col < side)
        if col < side
            push!(sources, i); push!(destinations, i + 1); push!(couplings, 1.0)
        end
        if row < side
            push!(sources, i); push!(destinations, i + side); push!(couplings, 1.0)
        end
    end
    return (; diagonal_entries, sources, destinations, couplings, b = sin.(1:n))
end

function compiled_grid_residual!(scratch, problem, means)
    @. scratch = problem.diagonal_entries * means - problem.b
    @inbounds for k in eachindex(problem.sources)
        i, j, a = problem.sources[k], problem.destinations[k], problem.couplings[k]
        scratch[i] -= a * means[j]
        scratch[j] -= a * means[i]
    end
    return norm(scratch, Inf) / norm(problem.b, Inf)
end

function compiled_grid_capacity(side, workers; max_sweeps = 5000, rss_limit = 8 * 1024^3)
    BLAS.set_num_threads(1)
    problem = compiled_grid_problem(side)
    n = length(problem.b)
    means, variances, previous_means, previous_vars, scratch = [zeros(n) for _ in 1:5]
    slots = Int32[]
    stable = Ref(0)
    residual = Ref(Inf)
    sweeps = Ref(0)
    callback = (;
        after_model_creation = event -> begin
            append!(slots, RxInfer.compiled_selected_slots(event.model, (x = KeepLast(),), false)[:x])
            @printf("BUILT side=%d workers=%d peak_RSS_GiB=%.4f\n", side, workers, Sys.maxrss()/1024^3)
            flush(stdout)
        end,
        after_iteration = event -> begin
            program = event.model.metadata[:inference_runner]
            settled = true
            @inbounds for i in eachindex(slots)
                posterior = getdata(program.values[slots[i]])
                means[i], variances[i] = mean(posterior), var(posterior)
                settled &= isapprox(means[i], previous_means[i]; rtol=1e-4, atol=1e-6)
                settled &= isapprox(variances[i], previous_vars[i]; rtol=1e-4, atol=1e-6)
            end
            residual[] = compiled_grid_residual!(scratch, problem, means)
            stable[] = settled ? stable[] + 1 : 0
            sweeps[] = event.iteration
            copyto!(previous_means, means)
            copyto!(previous_vars, variances)
            if event.iteration % 25 == 0 || (stable[] >= 5 && residual[] <= 1e-6)
                @printf("SWEEP side=%d sweep=%d residual=%.4e stable=%d peak_RSS_GiB=%.4f\n",
                    side, event.iteration, residual[], stable[], Sys.maxrss()/1024^3)
                flush(stdout)
            end
            Sys.maxrss() < rss_limit || error("Peak RSS capacity gate exceeded; terminating this probe")
            event.stop_iteration = stable[] >= 5 && residual[] <= 1e-6
        end,
    )
    (; diagonal_entries, sources, destinations, couplings, b) = problem
    result = infer(; model = compiled_edge_linear_system(; diagonal_entries, sources, destinations, couplings),
        data = (; b), initialization = linear_initialization, iterations = max_sweeps,
        options = (runner = CompiledRunner(; workers),), returnvars = (x = KeepLast(),),
        callbacks = callback, session = nothing, disable_inference_error_hint = true)
    GC.gc(true)
    peak = Sys.maxrss()
    success = residual[] <= 1e-6 && stable[] >= 5 && peak < rss_limit && all(>(0), variances)
    @printf("CAPACITY side=%d variables=%d workers=%d sweeps=%d residual=%.4e stable=%d peak_RSS_GiB=%.4f live_heap_GiB=%.4f pass=%s\n",
        side, n, workers, sweeps[], residual[], stable[], peak/1024^3, Base.gc_live_bytes()/1024^3, success)
    @assert length(result.posteriors[:x]) == n
    @assert success
    return result
end

function compiled_capacity_main()
    done = Threads.Atomic{Bool}(false)
    limit = 8 * 1024^3
    # Best-effort watchdog, not an OS-enforced memory cap. It also covers model
    # construction before inference callbacks become available.
    watchdog = Threads.@spawn begin
        while !done[]
            if Sys.maxrss() >= limit
                println(stderr, "CAPACITY_ABORT peak_RSS_GiB=", Sys.maxrss()/1024^3, " limit_GiB=8")
                flush(stderr)
                exit(2)
            end
            sleep(0.25)
        end
    end
    try
        compiled_grid_capacity(isempty(ARGS) ? 64 : parse(Int, ARGS[1]), length(ARGS) < 2 ? Threads.nthreads() : parse(Int, ARGS[2]))
    finally
        done[] = true
        wait(watchdog)
    end
end
abspath(PROGRAM_FILE) == (@__FILE__) && compiled_capacity_main()
