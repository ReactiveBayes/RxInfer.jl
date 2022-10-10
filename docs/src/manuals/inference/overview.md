# [Inference execution](@id user-guide-inference-execution)

The `RxInfer` inference API supports different types of message-passing algorithms (including hybrid algorithms combining several different types):

- [Belief Propagation](https://en.wikipedia.org/wiki/Belief_propagation)
- [Variational Message Passing](https://en.wikipedia.org/wiki/Variational_message_passing)

Whereas belief propagation computes exact inference for the random variables of interest, the variational message passing (VMP) in an approximation method that can be applied to a larger range of models.

The inference engine itself isn't aware of different algorithm types and simply does message passing between nodes, however during model specification stage user may specify different factorisation constraints around factor nodes by using `where { q = ... }` syntax or with the help of the `@constraints` macro. Different factorisation constraints lead to a different message passing update rules. See more documentation about constraints specification in the corresponding [section](@ref user-guide-constraints-specification).

## [Automatic inference specification on static datasets](@id user-guide-inference-execution-automatic-specification-static)

`RxInfer` exports the `inference` function to quickly run and test you model with static datasets. See more information about the `inference` function on the separate [documentation section](@ref user-guide-inference). 

## [Automatic inference specification on real-time datasets](@id user-guide-inference-execution-automatic-specification-realtime)

`RxInfer` exports the `rxinference` function to quickly run and test you model with dynamic and potentially real-time datasets. See more information about the `rxinference` function on the separate [documentation section](@ref user-guide-rxinference). 

## [Manual inference specification](@id user-guide-inference-execution-manual-specification)

While both `inference` and `rxinference` use most of the `RxInfer` inference engine capabilities in some situations it might be beneficial to write inference code manually. The [Manual inference](@ref user-guide-manual-inference) documentation section explains how to write your custom inference routines.