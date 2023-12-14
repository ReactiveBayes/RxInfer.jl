# [Automatic Inference Specification](@id user-guide-inference)

`RxInfer` provides the `infer` function for quickly running and testing your model with both static and streaming datasets. To enable streaming behavior, the `infer` function accepts an `autoupdates` argument, which specifies how to update your priors for future states based on newly updated posteriors. 

It's important to note that while this function covers most capabilities of the inference engine, advanced use cases may require resorting to the [Manual Inference Specification](@ref user-guide-inference-execution-manual-specification).

For details on manual inference specification, see the [Manual Inference](@ref user-guide-manual-inference) section.

```@docs
infer
InferenceResult
RxInfer.start
RxInfer.stop
@autoupdates
RxInferenceEngine
RxInferenceEvent
```