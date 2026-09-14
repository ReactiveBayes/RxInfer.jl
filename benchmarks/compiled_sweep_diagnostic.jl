# Isolates the executable-program sweep for optimization diagnostics. This is
# NOT an end-to-end or accuracy-matched comparison against the reactive backend.
using BenchmarkTools, Printf
include("compiled_grid_capacity.jl")

function sweep_fixture(problem, workers, optimize)
    (; diagonal_entries, sources, destinations, couplings, b) = problem
    result = infer(; model = compiled_edge_linear_system(; diagonal_entries, sources, destinations, couplings),
        data = (; b), initialization = linear_initialization, iterations = 10,
        options = (runner = CompiledRunner(; workers, optimize),), returnvars = (x = KeepLast(),),
        session = nothing, disable_inference_error_hint = true)
    return result.model.metadata[:inference_runner], mean.(result.posteriors[:x])
end

function sweep_diagnostic(side)
    BLAS.set_num_threads(1)
    problem = compiled_grid_problem(side)
    configs = [(1, false), (1, true), (Threads.nthreads(), true)]
    fixtures = [sweep_fixture(problem, workers, optimize) for (workers, optimize) in configs]
    @assert all(fixture -> fixture[2] == first(fixtures)[2], fixtures)
    for (program, _) in fixtures
        ReactiveMP.compiled_sweep!(program)
    end
    for round in 1:2
        order = round == 1 ? eachindex(configs) : reverse(eachindex(configs))
        for i in order
            program = fixtures[i][1]
            GC.gc(true)
            trial = @benchmark ReactiveMP.compiled_sweep!($program) samples=3 evals=1 seconds=120
            estimate = median(trial)
            @printf("SWEEP_BT side=%d workers=%d optimize=%s round=%d median_ms=%.6f bytes=%d allocations=%d gc_ms=%.6f typed=%s\n",
                side, configs[i][1], configs[i][2], round, estimate.time/1e6,
                estimate.memory, estimate.allocs, estimate.gctime/1e6, program.typed !== nothing)
            flush(stdout)
        end
    end
end
abspath(PROGRAM_FILE) == (@__FILE__) && sweep_diagnostic(isempty(ARGS) ? 64 : parse(Int, ARGS[1]))
