# [Inference execution](@id user-guide-inference-execution)

The `RxInfer` inference API supports different types of message-passing algorithms (including hybrid algorithms combining several different types):

- [Belief Propagation](https://en.wikipedia.org/wiki/Belief_propagation)
- [Variational Message Passing](https://en.wikipedia.org/wiki/Variational_message_passing)

Whereas belief propagation computes exact inference for the random variables of interest, the variational message passing (VMP) in an approximation method that can be applied to a larger range of models.

The inference engine itself isn't aware of different algorithm types and simply does message passing between nodes, however during model specification stage user may specify different factorisation constraints around factor nodes with the help of the [`@constraints`](@ref user-guide-constraints-specification) macro. Different factorisation constraints lead to a different message passing update rules. See more documentation about constraints specification in the corresponding [section](@ref user-guide-constraints-specification).

## [Automatic inference specification](@id user-guide-inference-execution-automatic-specification)

`RxInfer` exports the [`infer`](@ref) function to quickly run and test you model with both static and asynchronous (real-time) datasets. See more information about the `infer` function on the separate documentation section:

- [Static Inference](@ref manual-static-inference). 
- [Streamlined Inference](@ref manual-online-inference). 

```@docs
infer
RxInfer.ReactiveMPInferenceOptions
```

