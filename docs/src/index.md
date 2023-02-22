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

Many important AI applications, such as audio-processing, self-driving vehicles, weather forecasting, extended reality video processing, and others require continually solving an inference task in sophisticated probabilistic models with a large number of latent variables. 
Often, the inference task in these applications must be performed continually and in real time in response to new observations. 
Popular MC-based inference methods, such as No U-Turn Samples (NUTS) or Hamiltonian Monte Carlo (HMC), rely on computationally heavy sampling procedures that do not scale well to probabilistic models with thousands of latent states. 
Therefore, MC-based inference is practically not suitable for real-time applications. 
While the alternative variational inference (VI) method promises to scale better to large models than sampling-based inference, VI requires the derivation of gradients of the "variational Free Energy" cost function. 
For large models, manual derivation of these gradients might be not feasible, while automated "black-box" gradient methods do not scale either because they are not capable of taking advantage of sparsity or conjugate pairs in the model. 
Therefore, while Bayesian inference is known as the optimal data processing framework, in practice, real-time AI applications rely on much simpler, often ad hoc, data processing algorithms.

`RxInfer` provides utility to run efficient Bayesian inference in sophisticated probabilistic models, 
takes advantages of conjugate relationships in probabilistic models, and focuses to perform real-time Bayesian inference 
in large state-space models with thousands of latent variables. In addition, `RxInfer` provides a straightforward 
way to extend its functionality with custom factor nodes and message passing update rules. The engine is capable of running
various Bayesian inference algorithms in different parts of the factor graph of a single probabilistic model. This makes it easier 
to explore different "what-if" scenarious and enables very efficient inference in specific cases.

## Ecosystem

The `RxInfer` unites 3 core packages into one powerful reactive message passing-based Bayesian inference framework:

- [`ReactiveMP.jl`](https://github.com/biaslab/ReactiveMP.jl) - core package for efficient and scalable for reactive message passing 
- [`GraphPPL.jl`](https://github.com/biaslab/GraphPPL.jl) - package for model and constraints specification
- [`Rocket.jl`](https://github.com/biaslab/Rocket.jl) - reactive programming tools

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
  "examples/overview.md",
  "library/exported-methods.md",
  "library/functional-forms.md",
  "contributing/overview.md",
  "contributing/new-example.md"
]
Depth = 2
```

## Resources

- For an introduction to message passing and FFGs, see [The Factor Graph Approach to Model-Based Signal Processing](https://ieeexplore.ieee.org/document/4282128/) by Loeliger et al. (2007).

## Index

```@index
```
