# Posterior compatibility probe for the actual notebook, without plotting.
# This is a correctness diagnostic, not a timing benchmark.
ENV["LOG_USING_RXINFER"] = "false"
using RxInfer, LinearAlgebra, Random, Statistics, Test
import JSON
const COMPILED_RSLDS_OPTIONS = Ref{Any}((limit_stack_depth = 100,))
const rslds_path = normpath(joinpath(@__DIR__, "../../RxInferExamples.jl/examples/Experimental Examples/Recurrent Switching Linear Dynamical System/Recurrent Switching Linear Dynamical System.ipynb"))
const rslds_notebook = JSON.parsefile(rslds_path)
const rslds_horizon = isempty(ARGS) ? 12 : parse(Int, ARGS[1])
for index in 2:5
    source = join(rslds_notebook["cells"][index]["source"])
    source = replace(source, "using StableRNGs" => "", "StableRNG(" => "MersenneTwister(",
        "T = 500" => "T = $rslds_horizon", "options = (limit_stack_depth = 100,)" => "options = COMPILED_RSLDS_OPTIONS[]",
        "returnvars = KeepEach()," => "returnvars = KeepLast(),",
        "free_energy = true," => "free_energy = true, session = nothing, disable_inference_error_hint = true,")
    include_string(Main, source, rslds_path * ":cell$index")
end
function rslds_check()
    iterations = length(ARGS) < 2 ? 1000 : parse(Int, ARGS[2])
    baseline = fit_rslds(y, 2, 2, 2; iterations, hyperparameters)
    println("RSLDS_REFERENCE free_energy=", baseline.free_energy[end])
    flush(stdout)
    COMPILED_RSLDS_OPTIONS[] = (runner = CompiledRunner(),)
    actual = fit_rslds(y, 2, 2, 2; iterations, hyperparameters)
    println("RSLDS_COMPILED free_energy=", actual.free_energy[end])
    @testset "RSLDS converged posterior agreement" begin
        for name in keys(baseline.posteriors), statistic in (mean, var)
            b, a = baseline.posteriors[name], actual.posteriors[name]
            bs = b isa AbstractArray ? statistic.(b) : statistic(b)
            as = a isa AbstractArray ? statistic.(a) : statistic(a)
            agrees = isapprox(as, bs; rtol=1e-4, atol=1e-6)
            println("RSLDS_POSTERIOR ", name, " statistic=", statistic,
                " agreement=", agrees, " relative_error=", norm(as - bs) / max(norm(bs), 1e-6))
            @test agrees
        end
    end
    return baseline, actual
end
rslds_check()
