# [Inference results postprocessing](@id user-guide-inference-postprocess)

[`infer`](@ref) allow users to postprocess the inference result with the `postprocess = ...` keyword argument. The inference engine 
operates on __wrapper__ types to distinguish between marginals and messages. By default these wrapper types are removed from the inference results if no addons option is present. Together with the enabled addons, however, the wrapper types are preserved in the inference result output value. Use the options below to change this behaviour:

```@docs
inference_postprocess
DefaultPostprocess
UnpackMarginalPostprocess
NoopPostprocess
```

## [Custom postprocessing step](@id user-guide-inference-postprocess)

In order to implement a custom postprocessing strategy simply implement the [`inference_postprocess`](@ref) method:

```@example custom-postprocessing
using RxInfer

struct CustomPostprocess end

# For demonstration purposes out postprocessing step simply stringifyes the result
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
