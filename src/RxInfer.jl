module RxInfer

using Reexport

@reexport using ReactiveMP, GraphPPL, Rocket, Distributions, ExponentialFamily, BayesBase, FastCholesky

include("helpers.jl")
include("rocket.jl")

include("score/actor.jl")
include("score/diagnostics.jl")

include("model/model.jl")
include("model/plugins/reactivemp_inference.jl")
include("model/plugins/reactivemp_free_energy.jl")
include("model/plugins/initialization_plugin.jl")
include("model/graphppl.jl")

include("constraints/form/form_fixed_marginal.jl")
include("constraints/form/form_point_mass.jl")
include("constraints/form/form_sample_list.jl")

include("inference/inference.jl")

end
