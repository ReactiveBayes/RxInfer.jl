# julia --project=RxInfer.jl/benchmarks --threads=6 RxInfer.jl/benchmarks/multicore_scaling.jl [dimension] [plate] [iterations]
using RxInfer, LinearAlgebra, Statistics, Printf

include("multicore_benchmark_tools.jl")

@model function multicore_scaling_model(y, dimension)
    p ~ GammaShapeRate(1.0, 1.0)
    for i in eachindex(y)
        x[i] ~ MvNormalMeanPrecision(zeros(dimension), diageye(dimension))
        y[i] ~ MvNormalMeanScalePrecision(x[i], p)
    end
end

const scaling_constraints = @constraints begin
    q(x, p) = q(x)q(p)
end
const scaling_initialization = @initialization begin
    q(p) = GammaShapeRate(1.0, 1.0)
end

function scaling_run(dimension, plate, iterations, workers)
    options = if workers == 0
        (limit_stack_depth = 100,)
    else
        (runner = MulticoreRunner(; workers), limit_stack_depth = 100)
    end
    return infer(;
        model = multicore_scaling_model(; dimension),
        data = (y = [ones(dimension) for _ in 1:plate],),
        constraints = scaling_constraints,
        initialization = scaling_initialization,
        iterations = iterations,
        returnvars = (p = KeepLast(),),
        options = options,
        session = nothing,
        disable_inference_error_hint = true,
    )
end

function main()
    dimension = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
    plate = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 256
    iterations = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20
    println(
        "model=isotropic_multivariate dimension=",
        dimension,
        " plate=",
        plate,
        " iterations=",
        iterations,
    )
    reference = nothing
    function check_result(workers, result)
        posterior = result.posteriors[:p]
        if isnothing(reference)
            reference = posterior
        else
            @assert isapprox(mean(posterior), mean(reference); rtol = 1e-10)
            @assert isapprox(var(posterior), var(reference); rtol = 1e-10)
        end
        @printf("CHECK workers=%d p=%.12f\n", workers, mean(posterior))
        state = ReactiveMP.find_multicore_runner(
            get(result.model.metadata, :inference_runner, nothing)
        )
        if !isnothing(state)
            println(
                "  waves=",
                state.waves,
                " parallel_jobs=",
                state.parallel_jobs,
                " serial_jobs=",
                state.serial_jobs,
            )
        end
        return nothing
    end
    return benchmark_multicore(
        w -> scaling_run(dimension, plate, iterations, w), check_result
    )
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
