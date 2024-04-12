# [Static Inference](@id manual-static-inference)

This guide explains how to use the [`infer`](@ref) function for static datasets. We'll show how `RxInfer` can estimate posterior beliefs given a set of observations. We'll use a simple Beta-Bernoulli model as an example, which has been covered in the [Getting Started](@ref user-guide-getting-started) section, but keep in mind that these techniques can apply to any model.

Also read about [Streaming Inference](@ref manual-online-inference) or checkout more complex [examples](https://reactivebayes.github.io/RxInfer.jl/stable/examples/overview/).

## [Model specification](@id manual-static-inference-model-spec)

Also read the [Model Specification](@ref user-guide-model-specification) section.

In static inference, we want to update our prior beliefs about certain hidden states given some dataset.
To achieve this, we include data as an argument in our model specification:

```@example manual-static-inference
using RxInfer
using Test #hide
```

```@docs
InferenceResult
```