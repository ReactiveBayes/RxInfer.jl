# julia --project=RxInfer.jl/benchmarks --threads=6 RxInfer.jl/benchmarks/multicore_grid.jl [side] [iterations]
using RxInfer, LinearAlgebra, Statistics
include("multicore_benchmark_tools.jl")
@model function multicore_grid_benchmark(b, side)
    for i in eachindex(b)
        x[i] ~ NormalMeanPrecision(b[i] / 5, 5.0)
    end
    for row in 1:side, col in 1:side
        i = (row - 1) * side + col
        if row < side
            x[i + side] ~ GaussianCoupling(x[i], 1.0)
        end
        if col < side
            x[i + 1] ~ GaussianCoupling(x[i], 1.0)
        end
    end
end
const grid_initialization = @initialization begin
    μ(x) = NormalMeanVariance(0.0, 1e6)
end
function grid_run(side, iterations, workers)
    options = if workers == 0
        (limit_stack_depth = 100,)
    else
        (runner = MulticoreRunner(; workers), limit_stack_depth = 100)
    end
    infer(;
        model = multicore_grid_benchmark(; side),
        data = (b = sin.(1:(side ^ 2)),),
        initialization = grid_initialization,
        iterations = iterations,
        options = options,
        returnvars = (x = KeepLast(),),
        session = nothing,
        disable_inference_error_hint = true,
    )
end
function main()
    side = isempty(ARGS) ? 32 : parse(Int, ARGS[1])
    iterations = length(ARGS) < 2 ? 40 : parse(Int, ARGS[2])
    println("model=diagonal5_grid side=", side, " iterations=", iterations)
    reference = nothing
    one = nothing
    function check_result(workers, result)
        if workers == 0
            reference = result.posteriors[:x]
        elseif workers == 1
            one = result.posteriors
        else
            @assert one == result.posteriors
        end
        @assert isapprox(
            mean.(reference), mean.(result.posteriors[:x]); atol = 1e-7
        )
        @assert isapprox(
            var.(reference), var.(result.posteriors[:x]); atol = 1e-7
        )
        state = get(result.model.metadata, :inference_runner, nothing)
        if !isnothing(state)
            println(
                "  waves=",
                state.waves,
                " parallel=",
                state.parallel_jobs,
                " serial=",
                state.serial_jobs,
            )
        end
        return nothing
    end
    return benchmark_multicore(w -> grid_run(side, iterations, w), check_result)
end
abspath(PROGRAM_FILE) == (@__FILE__) && main()
