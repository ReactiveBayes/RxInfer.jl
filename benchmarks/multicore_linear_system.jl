# julia --project=RxInfer.jl/benchmarks --threads=6 RxInfer.jl/benchmarks/multicore_linear_system.jl [side] [iterations] [leak]
using RxInfer, LinearAlgebra, SparseArrays
import JSON
include("multicore_benchmark_tools.jl")

# Load the actual notebook model, including its dense upper-triangle scan.
# Do not load its plotting, simulation, or large heat-grid state-space cells.
const linear_notebook_path = normpath(
    joinpath(
        @__DIR__,
        "../../RxInferExamples.jl/examples/Advanced Examples/Solving Linear Systems with Message Passing/Solving Linear Systems with Message Passing.ipynb",
    ),
)
const linear_notebook = JSON.parsefile(linear_notebook_path)
const linear_model_source = only(
    filter(
        source -> occursin("@model function linear_system_model(b, A)", source),
        [join(cell["source"]) for cell in linear_notebook["cells"]],
    ),
)
include_string(
    Main, linear_model_source, linear_notebook_path * ":linear_system_model"
)

const linear_initialization = @initialization begin
    μ(x) = NormalMeanVariance(0.0, 1e6)
end

# Same degree + leak diagonal and -1 neighbor entries as grid_laplacian
# in the notebook; avoid its additional full-size A + leak * I allocation.
function linear_grid_matrix(side, leak)
    side >= 2 || throw(ArgumentError("side must be at least two"))
    leak > 0 || throw(ArgumentError("leak must be positive"))
    A = zeros(side^2, side^2)
    index(r, c) = (r - 1) * side + c
    for r in 1:side, c in 1:side
        i = index(r, c)
        A[i, i] = leak
        for (dr, dc) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            rr, cc = r + dr, c + dc
            if 1 <= rr <= side && 1 <= cc <= side
                A[i, index(rr, cc)] = -1.0
                A[i, i] += 1.0
            end
        end
    end
    return A
end

function linear_run(A, b, iterations, workers)
    options = if workers == 0
        (limit_stack_depth = 100,)
    else
        (runner = MulticoreRunner(; workers), limit_stack_depth = 100)
    end
    return infer(;
        model = linear_system_model(; A),
        data = (; b),
        initialization = linear_initialization,
        iterations,
        options,
        returnvars = (x = KeepLast(),),
        session = nothing,
        disable_inference_error_hint = true,
    )
end

function main()
    side = isempty(ARGS) ? 20 : parse(Int, ARGS[1])
    iterations = length(ARGS) < 2 ? 100 : parse(Int, ARGS[2])
    leak = length(ARGS) < 3 ? 0.1 : parse(Float64, ARGS[3])
    A = linear_grid_matrix(side, leak)
    b = sin.(1:(side ^ 2))
    sparse_A = sparse(A)
    exact = sparse_A \ b
    println(
        "model=notebook_linear_system side=",
        side,
        " leak=",
        leak,
        " iterations=",
        iterations,
        " RHS=sin.(1:side^2) returnvars=KeepLast",
    )
    println(
        "Matrix/RHS generation, direct solve, accuracy checks, SPD checks and plotting excluded from timing.",
    )
    references = Dict{Int, Any}()
    function check_result(workers, result)
        means = mean.(result.posteriors[:x])
        variances = var.(result.posteriors[:x])
        if haskey(references, workers)
            @assert means == references[workers][1]
            @assert variances == references[workers][2]
        else
            references[workers] = (means, variances)
        end
        if workers > 1
            @assert means == references[1][1]
            @assert variances == references[1][2]
        end
        residual = norm(sparse_A * means - b, Inf) / norm(b, Inf)
        mean_error = norm(means - exact, Inf) / norm(exact, Inf)
        @printf(
            "CHECK workers=%d relative_residual=%.6e relative_mean_error=%.6e\n",
            workers,
            residual,
            mean_error
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
    trials = benchmark_multicore(
        w -> linear_run(A, b, iterations, w), check_result
    )
    println(
        "Fixed iteration comparison, NOT a time-to-equal-accuracy benchmark."
    )
    return trials
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
