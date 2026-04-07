export inference_postprocess,
    UnpackMarginalPostprocess, NoopPostprocess

"""
    inference_postprocess(strategy, result)

Postprocesses the `result` of the inference procedure according to the given `strategy`.
The `result` is typically a `Marginal` or a collection of `Marginal`s.

The [`infer`](@ref) function selects a default strategy automatically based on whether
annotations are enabled:
- When `annotations = nothing` (the default), the strategy is [`UnpackMarginalPostprocess`](@ref),
  which strips the `Marginal` wrapper and returns the underlying distribution directly.
- When annotations are provided, the strategy is [`NoopPostprocess`](@ref), which preserves
  the `Marginal` wrapper so that annotation data remains accessible via `ReactiveMP.getannotations`.

You can override this default by passing a custom `postprocess` argument to [`infer`](@ref).
See [Inference results postprocessing](@ref user-guide-inference-postprocess) for more details.
"""
function inference_postprocess end

"""
    UnpackMarginalPostprocess

A postprocessing strategy that removes the `Marginal` wrapper type from the inference result,
returning the underlying distribution directly. This is the default strategy when no
annotations are enabled, since the `Marginal` wrapper carries no extra information in that case.

See also: [`NoopPostprocess`](@ref), [`inference_postprocess`](@ref)
"""
struct UnpackMarginalPostprocess end

inference_postprocess(::UnpackMarginalPostprocess, result::Marginal) = getdata(
    result
)
inference_postprocess(::UnpackMarginalPostprocess, result::AbstractArray) = map(
    (element) -> inference_postprocess(UnpackMarginalPostprocess(), element),
    result,
)

"""
    NoopPostprocess

A postprocessing strategy that preserves the inference result as-is, keeping the `Marginal`
wrapper intact. This is the default strategy when annotations are enabled, so that annotation
data attached to each marginal remains accessible via `ReactiveMP.getannotations`.

See also: [`UnpackMarginalPostprocess`](@ref), [`inference_postprocess`](@ref)
"""
struct NoopPostprocess end

inference_postprocess(::NoopPostprocess, result) = result
inference_postprocess(::Nothing, result) = result
