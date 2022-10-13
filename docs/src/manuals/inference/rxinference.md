# [Automatic inference specification on real-time datasets](@id user-guide-rxinference)

`RxInfer` exports the `rxinference` function to quickly run and test you model with dynamic and potentially real-time datasets. Note, however, that this function does cover almost all capabilities of the __reactive__ inference engine, but for advanced use cases you may want to resort to the [manual inference specification](@ref user-guide-inference-execution-manual-specification).

For running inference on static datasets see the [Static Inference](@ref user-guide-inference) section.
For manual inference specification see the [Manual Inference](@ref user-guide-manual-inference) section.

```@docs
@autoupdates
rxinference
RxInfer.start
RxInfer.stop
RxInferenceEngine
RxInferenceEvent
```