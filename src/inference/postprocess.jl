export inference_postprocess, DefaultPostprocess, UnpackMarginalPostprocess, NoopPostprocess

"""
    inference_postprocess(strategy, result)
  
This function modifies the `result` of the inference procedure according to the strategy. 
The `result` can be a `Marginal` or a collection of `Marginal`s.
The default `strategy` is [`DefaultPostprocess`](@ref).
"""
function inference_postprocess end

"""`DefaultPostprocess` picks the most suitable postprocessing step automatically."""
struct DefaultPostprocess end

inference_postprocess(::DefaultPostprocess, result::Marginal) = inference_postprocess(DefaultPostprocess(), result, ReactiveMP.getaddons(result))
inference_postprocess(::DefaultPostprocess, result::AbstractArray) = map((element) -> inference_postprocess(DefaultPostprocess(), element), result)

# Default postprocessing step removes Marginal type wrapper if no addons are present, and keeps the Marginal type wrapper otherwise
inference_postprocess(::DefaultPostprocess, result, addons::Nothing) = inference_postprocess(UnpackMarginalPostprocess(), result)
inference_postprocess(::DefaultPostprocess, result, addons::Any) = inference_postprocess(NoopPostprocess(), result)

"""This postprocessing step removes the `Marginal` wrapper type from the result."""
struct UnpackMarginalPostprocess end

inference_postprocess(::UnpackMarginalPostprocess, result::Marginal) = getdata(result)
inference_postprocess(::UnpackMarginalPostprocess, result::AbstractArray) = map((element) -> inference_postprocess(UnpackMarginalPostprocess(), element), result)

"""This postprocessing step does nothing."""
struct NoopPostprocess end

inference_postprocess(::NoopPostprocess, result) = result
inference_postprocess(::Nothing, result) = result