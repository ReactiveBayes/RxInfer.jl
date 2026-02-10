module RxInfer

using Reexport

@reexport using ReactiveMP, GraphPPL, Rocket, Distributions, ExponentialFamily, BayesBase, FastCholesky

include("helpers.jl")
include("rocket.jl")
include("session.jl")
include("telemetry.jl")

include("score/actor.jl")
include("score/diagnostics.jl")

include("model/model.jl")
include("model/graphppl.jl")
include("model/plugins/reactivemp_inference.jl")
include("model/plugins/reactivemp_free_energy.jl")
include("model/plugins/reactivemp_force_marginal_computation_plugin.jl")
include("model/plugins/initialization_plugin.jl")

include("constraints/form/form_ensure_supported.jl")
include("constraints/form/form_fixed_marginal.jl")
include("constraints/form/form_point_mass.jl")
include("constraints/form/form_sample_list.jl")

include("inference/postprocess.jl")
include("inference/benchmarkcallbacks.jl")
include("inference/inference.jl")

include("callbacks/stop_early.jl")

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
