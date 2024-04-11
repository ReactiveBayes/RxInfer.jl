# [Automatic Inference Specification](@id user-guide-inference)

`RxInfer` provides the `infer` function for quickly running and testing your model with both static and streaming datasets. To enable streaming behavior, the `infer` function accepts an `autoupdates` argument, which specifies how to update your priors for future states based on newly updated posteriors. 

```@docs
infer
InferenceResult
RxInfer.ReactiveMPInferenceOptions
```