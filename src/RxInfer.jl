module RxInfer

using Reexport, PrecompileTools

@recompile_invalidations begin
    @reexport using ReactiveMP, GraphPPL, Rocket, Distributions, ExponentialFamily, BayesBase, FastCholesky
end

include("helpers.jl")
include("rocket.jl")
include("session.jl")
include("telemetry.jl")

include("score/actor.jl")
include("score/diagnostics.jl")

include("model/model.jl")
include("model/plugins/reactivemp_inference.jl")
include("model/plugins/reactivemp_free_energy.jl")
include("model/plugins/reactivemp_force_marginal_computation_plugin.jl")
include("model/plugins/initialization_plugin.jl")
include("model/graphppl.jl")

include("constraints/form/form_ensure_supported.jl")
include("constraints/form/form_fixed_marginal.jl")
include("constraints/form/form_point_mass.jl")
include("constraints/form/form_sample_list.jl")

include("inference/postprocess.jl")
include("inference/benchmarkcallbacks.jl")
include("inference/inference.jl")

# A simple precompile workload to trigger compilation of the infer function
@setup_workload begin
    distribution = Bernoulli(0.5)
    dataset      = rand(distribution, 10)

    @model function coin_model(y, a, b)
        θ ~ Beta(a, b)
        for i in eachindex(y)
            y[i] ~ Bernoulli(θ)
        end
    end

    @compile_workload begin
        infer(model = coin_model(a = 2.0, b = 7.0), data = (y = dataset,))
    end
end

_isprecompiling() = ccall(:jl_generating_output, Cint, ()) == 1

function __init__()
    if !_isprecompiling()
        if RxInfer.preference_enable_session_logging
            default_session = create_session()
            RxInfer.set_default_session!(default_session)
        end

        if RxInfer.preference_enable_using_rxinfer_telemetry
            log_using_rxinfer()
        end
    end
end

end
