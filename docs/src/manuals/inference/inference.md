# [Automatic inference specification on static datasets](@id user-guide-inference)

`RxInfer` exports the `inference` function to quickly run and test you model with static datasets. Note, however, that this function does cover almost all capabilities of the inference engine, but for advanced use cases you may want to resort to the [manual inference specification](@ref user-guide-inference-execution-manual-specification). 

For running inference on real-time datasets see the [Reactive Inference](@ref user-guide-rxinference) section.
For manual inference specification see the [Manual Inference](@ref user-guide-manual-inference) section.

```@docs
inference
InferenceResult
```