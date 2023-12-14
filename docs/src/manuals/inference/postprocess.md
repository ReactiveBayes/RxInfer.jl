# [Inference results postprocessing](@id user-guide-inference-postprocess)

[`infer`](@ref) allow users to postprocess the inference result with the `postprocess = ...` keyword argument. The inference engine 
operates on __wrapper__ types to distinguish between marginals and messages. By default 
these wrapper types are removed from the inference results if no addons option is present.
Together with the enabled addons, however, the wrapper types are preserved in the 
inference result output value. Use the options below to change this behaviour:

```@docs
DefaultPostprocess
UnpackMarginalPostprocess
NoopPostprocess
```
