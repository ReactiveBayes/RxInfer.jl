```@meta
CurrentModule = RxInfer
```

RxInfer
=======

```@raw html
<div class="light-biglogo">
```
![RxInfer Logo](assets/biglogo.svg)
```@raw html
</div>
```

```@raw html
<div class="dark-biglogo">
```
![RxInfer Logo](assets/biglogo-blacktheme.svg)
```@raw html
</div>
```

*Julia package for automatic Bayesian inference on a factor graph with reactive message passing.*

Given a probabilistic model, RxInfer allows for an efficient message-passing based Bayesian inference. It uses the model structure to generate an algorithm that consists of a sequence of local computations on a Forney-style factor graph (FFG) representation of the model. RxInfer.jl has been designed with a focus on efficiency, scalability and maximum performance for running inference with message passing.

## Package Features

- User friendly syntax for specification of probabilistic models.
- Automatic generation of message passing algorithms including
    - [Belief propagation](https://en.wikipedia.org/wiki/Belief_propagation)
    - [Variational message passing](https://en.wikipedia.org/wiki/Variational_message_passing)
    - [Expectation maximization](https://en.wikipedia.org/wiki/Expectation-maximization_algorithm)
- Support for hybrid models combining discrete and continuous latent variables.
- Support for hybrid distinct message passing inference algorithm under a unified paradigm.
- Factorisation and functional form constraints specification.
- Evaluation of Bethe free energy as a model performance measure.
- Schedule-free reactive message passing API.
- High performance.
- Scalability for large models with millions of parameters and observations.
- Inference procedure is differentiable.
- Easy to extend with custom nodes and message update rules.

## Why RxInfer

Many important AI applications, including audio processing, self-driving vehicles, weather forecasting, and extended-reality video processing require continually solving an inference task in sophisticated probabilistic models with a large number of latent variables.
Often, the inference task in these applications must be performed continually and in real-time in response to new observations.
Popular MC-based inference methods, such as the No U-Turn Sampler (NUTS) or Hamiltonian Monte Carlo (HMC) sampling, rely on computationally heavy sampling procedures that do not scale well to probabilistic models with thousands of latent states.
Therefore, MC-based inference is practically not suitable for real-time applications.
While the alternative variational inference method (VI) promises to scale better to large models than sampling-based inference, VI requires the derivation of gradients of a "variational Free Energy" cost function.
For large models, manual derivation of these gradients might not be feasible, while automated "black-box" gradient methods do not scale either because they are not capable of taking advantage of sparsity or conjugate pairs in the model.
Therefore, while Bayesian inference is known as the optimal data processing framework, in practice, real-time AI applications rely on much simpler, often ad hoc, data processing algorithms.

RxInfer aims to remedy these issues by running efficient Bayesian inference in sophisticated probabilistic models,
taking advantage of local conjugate relationships in probabilistic models, and focusing on real-time Bayesian inference in large state-space models with thousands of latent variables. In addition, RxInfer provides a straightforward way to extend its functionality with custom factor nodes and message passing update rules. The engine is capable of running
various Bayesian inference algorithms in different parts of the factor graph of a single probabilistic model. This makes it easier
to explore different "what-if" scenarios and enables very efficient inference in specific cases.

## How to get started?

Head to the [Getting started](@ref user-guide-getting-started) section to get up and running with RxInfer. Alternatively, explore various [examples](@ref examples-overview) in the documentation.

## Table of Contents

```@contents
Pages = [
  "manuals/background.md",
  "manuals/getting-started.md",
  "manuals/model-specification.md",
  "manuals/constraints-specification.md",
  "manuals/meta-specification.md",
  "manuals/inference-execution.md",
  "manuals/custom-node.md",
  "manuals/debugging.md",
  "manuals/delta-node.md",
  "examples/overview.md",
  "library/exported-methods.md",
  "library/functional-forms.md",
  "contributing/overview.md",
  "contributing/new-example.md"
]
Depth = 2
```

## References

- [RxInfer: A Julia package for reactive real-time Bayesian inference](https://doi.org/10.21105/joss.05161) - a reference paper for the `RxInfer.jl` framwork.
- [Variational Message Passing and Local Constraint Manipulation in Factor Graphs](https://doi.org/10.3390/e23070807) - describes theoretical aspects of the underlying Bayesian inference method.
- [Reactive Message Passing for Scalable Bayesian Inference](https://doi.org/10.48550/arXiv.2112.13251) - describes implementation aspects of the Bayesian inference engine and performs benchmarks and accuracy comparison on various models.
- [A Julia package for reactive variational Bayesian inference](https://doi.org/10.1016/j.simpa.2022.100299) - a reference paper for the `ReactiveMP.jl` package, the underlying inference engine.
- [The Factor Graph Approach to Model-Based Signal Processing](https://ieeexplore.ieee.org/document/4282128/) - an introduction to message passing and FFGs.

## Ecosystem

The `RxInfer` unites 3 core packages into one powerful reactive message passing-based Bayesian inference framework:

- [`ReactiveMP.jl`](https://github.com/biaslab/ReactiveMP.jl) - core package for efficient and scalable for reactive message passing 
- [`GraphPPL.jl`](https://github.com/biaslab/GraphPPL.jl) - package for model and constraints specification
- [`Rocket.jl`](https://github.com/biaslab/Rocket.jl) - reactive programming tools

## Index

```@index
```
## Bibliography
```@bibliography
```