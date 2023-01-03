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
