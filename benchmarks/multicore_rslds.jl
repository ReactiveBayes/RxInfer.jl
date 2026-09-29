# Runs the existing RSLDS notebook's model and custom rules without plotting.
# julia --project=RxInfer.jl/benchmarks --threads=6 RxInfer.jl/benchmarks/multicore_rslds.jl [T] [iterations] [samples_per_round]
using RxInfer, LinearAlgebra, Random, Statistics
import JSON

include("multicore_benchmark_tools.jl")
const RSLDS_OPTIONS = Ref{Any}((limit_stack_depth = 100,))
const notebook_path = normpath(
    joinpath(
        @__DIR__,
        "../../RxInferExamples.jl/examples/Experimental Examples/Recurrent Switching Linear Dynamical System/Recurrent Switching Linear Dynamical System.ipynb",
    ),
)
const notebook = JSON.parsefile(notebook_path)
const horizon = isempty(ARGS) ? 100 : parse(Int, ARGS[1])
const niterations = length(ARGS) < 2 ? 10 : parse(Int, ARGS[2])
const nrepeats = if length(ARGS) < 3
    benchmark_integer("RXINFER_BENCHMARK_SAMPLES", 5)
else
    parse(Int, ARGS[3])
end

for index in 2:5
    source = join(notebook["cells"][index]["source"])
    source = replace(
        source,
        "using StableRNGs" => "",
        "StableRNG(" => "MersenneTwister(",
        "T = 500" => "T = $horizon",
        "options = (limit_stack_depth = 100,)" => "options = RSLDS_OPTIONS[]",
        "free_energy = true," => "free_energy = true, session = nothing, disable_inference_error_hint = true,",
    )
    include_string(Main, source, notebook_path * ":cell$index")
end

serial_wave_posteriors = Ref{Any}(nothing)
serial_wave_free_energy = Ref{Any}(nothing)
# MixtureDistribution does not define value equality; compare its weights and
# components explicitly instead of using object identity for these posteriors.
same_rslds_posterior(a, b) = a == b
same_rslds_posterior(a::MixtureDistribution, b::MixtureDistribution) =
    weights(a) == weights(b) &&
    same_rslds_posterior(components(a), components(b))
same_rslds_posterior(a::AbstractArray, b::AbstractArray) =
    axes(a) == axes(b) &&
    all(same_rslds_posterior(x, y) for (x, y) in zip(a, b))
function rslds_run(workers)
    RSLDS_OPTIONS[] = if workers == 0
        (limit_stack_depth = 100,)
    else
        (runner = MulticoreRunner(; workers), limit_stack_depth = 100)
    end
    return fit_rslds(
        y, 2, 2, 2; iterations = niterations, hyperparameters = hyperparameters
    )
end

function check_rslds_result(workers, result)
    if workers == 1
        serial_wave_posteriors[] = result.posteriors
        serial_wave_free_energy[] = result.free_energy
    elseif workers > 1
        for key in keys(result.posteriors)
            @assert same_rslds_posterior(
                result.posteriors[key], serial_wave_posteriors[][key]
            ) "Posterior mismatch for $key with $workers workers"
        end
        @assert result.free_energy == serial_wave_free_energy[]
    end
    println("workers=", workers, " FE=", result.free_energy[end])
    println("  x[1]=", mean(result.posteriors[:x][end][1]))
    state = ReactiveMP.find_multicore_runner(
        get(result.model.metadata, :inference_runner, nothing)
    )
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

println("model=notebook_rslds horizon=", horizon, " iterations=", niterations)
benchmark_multicore(rslds_run, check_rslds_result; samples = nrepeats)
