# [Inference results postprocessing](@id user-guide-inference-postprocess)

The [`infer`](@ref) function allows users to control how inference results are postprocessed 
via the `postprocess` keyword argument.

Internally, the inference engine operates on `Marginal` wrapper types that carry both the 
underlying distribution and any associated annotation data. The postprocessing step determines
whether these wrappers are preserved or stripped from the final result.

## Default behavior

The default postprocessing strategy in [`infer`](@ref) depends on whether the `annotations = ` is specified or not:

- **Without annotations** (`annotations = nothing`, the default): the strategy is [`UnpackMarginalPostprocess`](@ref). 
  Since no annotation data is attached, the `Marginal` wrapper is removed and the result 
  contains the underlying distribution directly (e.g., a `Normal` or `Beta` distribution).
- **With annotations** (e.g., `annotations = LogScaleAnnotations()`): the strategy is [`NoopPostprocess`](@ref). 
  The `Marginal` wrapper is preserved so that annotation data remains accessible via 
  `ReactiveMP.getannotations`.

You can always override the default by passing `postprocess = ...` explicitly to [`infer`](@ref).

## Available strategies

```@docs
inference_postprocess
UnpackMarginalPostprocess
NoopPostprocess
```

## [Custom postprocessing step](@id user-guide-inference-postprocess-custom)

To implement a custom postprocessing strategy, define a new type and implement the 
[`inference_postprocess`](@ref) method for it:

```@example custom-postprocessing
using RxInfer

struct CustomPostprocess end

# For demonstration purposes our postprocessing step simply stringifies the result
RxInfer.inference_postprocess(::CustomPostprocess, result::Marginal) = string(ReactiveMP.getdata(result))
```

Now, we can use the postprocessing step in the [`infer`](@ref) function:

```@example custom-postprocessing
using Test #hide
@model function beta_bernoulli(y)
    θ ~ Beta(1, 1)
    y ~ Bernoulli(θ)
end

result = infer(
    model = beta_bernoulli(),
    data  = (y = 1.,),
    postprocess = CustomPostprocess()
)

@test occursin("Beta{Float64}(α=2.0, β=1.0)", result.posteriors[:θ]) #hide
result.posteriors[:θ] # should be a `String`
```

```@example custom-postprocessing
@test result.posteriors[:θ] isa String #hide
result.posteriors[:θ] isa String
```
