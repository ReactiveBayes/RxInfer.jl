```@meta
CurrentModule = RxInfer
```

RxInfer
=======

![RxInfer Logo](assets/biglogo.svg)

*Julia package for automatic Bayesian inference on a factor graph with reactive message passing.*

Given a probabilistic model, RxInfer allows for an efficient message-passing based Bayesian inference. It uses the model structure to generate an algorithm that consists of a sequence of local computations on a Forney-style factor graph (FFG) representation of the model. RxInfer.jl has been designed with a focus on efficiency, scalability and maximum performance for running inference with message passing.

See also:
- [`ReactiveMP.jl`](https://github.com/biaslab/ReactiveMP.jl)
- [`GraphPPL.jl`](https://github.com/biaslab/GraphPPL.jl)
- [`Rocket.jl`](https://github.com/biaslab/Rocket.jl)

## Package Features

- User friendly syntax for specification of probabilistic models.
- Automatic generation of message passing algorithms including
    - [Belief propagation](https://en.wikipedia.org/wiki/Belief_propagation)
    - [Variational message passing](https://en.wikipedia.org/wiki/Variational_message_passing)
    - [Expectation maximization](https://en.wikipedia.org/wiki/Expectation-maximization_algorithm)
- Support for hybrid models combining discrete and continuous latent variables.
- Support for hybrid distinct message passing inference algorithm under a unified paradigm.
- Evaluation of Bethe free energy as a model performance measure.
- Schedule-free reactive message passing API.
- High performance.
- Scalability for large models with millions of parameters and observations.
- Inference procedure is differentiable.
- Easy to extend with custom nodes.

## How to get started?
Head to the [Getting started](@ref user-guide-getting-started) section to get up and running with ForneyLab. Alternatively, explore various [examples](@ref examples-overview) in the documentation. For advanced extensive tutorial take a look on [Advanced Tutorial](@ref user-guide-advanced-tutorial).

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
