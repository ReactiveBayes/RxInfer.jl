module RxInfer

using Reexport

@reexport using ReactiveMP, GraphPPL, Rocket, Distributions

include("graphppl.jl")
include("model.jl")
include("inference.jl")

end
