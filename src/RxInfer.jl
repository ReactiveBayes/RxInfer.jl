module RxInfer

using Reexport

@reexport using ReactiveMP, GraphPPL, Rocket, Distributions, ExponentialFamily, BayesBase, FastCholesky

include("helpers.jl")
include("rocket.jl")
include("graphppl.jl")
include("model.jl")

include("compatibility/old_graphppl.jl")

include("constraints/form/form_fixed_marginal.jl")
include("constraints/form/form_point_mass.jl")
include("constraints/form/form_sample_list.jl")

include("score/actor.jl")
include("score/bfe.jl")

include("inference.jl")

end
