![](docs/src/assets/biglogo-blacktheme.svg?raw=true&sanitize=true)

[![Official page](https://img.shields.io/badge/official%20page%20-RxInfer-blue)](https://rxinfer.com)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://docs.rxinfer.com/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://docs.rxinfer.com/dev/)
[![Examples](https://img.shields.io/badge/examples-RxInfer-brightgreen)](https://examples.rxinfer.com/)
[![Q&A](https://img.shields.io/badge/Q&A-RxInfer-orange)](https://github.com/reactivebayes/RxInfer.jl/discussions)
[![Roadmap](https://img.shields.io/badge/roadmap-RxInfer-yellow)](#roadmap)
[![Build Status](https://github.com/reactivebayes/RxInfer.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/reactivebayes/RxInfer.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/reactivebayes/RxInfer.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/reactivebayes/RxInfer.jl)
[![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FRxInfer&query=total_requests&suffix=%2Fmonth&label=Downloads)](https://juliapkgstats.com/pkg/RxInfer)
[![DOI](https://img.shields.io/badge/Journal%20of%20Open%20Source%20Software-10.21105/joss.05161-critical)](https://doi.org/10.21105/joss.05161)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281/zenodo.7774921-important)](https://zenodo.org/badge/latestdoi/501995296)
[![NumFocus](https://img.shields.io/badge/supported%20by-NumFocus-green)](#numfocus)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ReactiveBayes/RxInfer.jl)
[![Join our Discord](https://img.shields.io/badge/Join%20our-Discord-blue)](https://discord.gg/UvMhrbQh)


# Overview

`RxInfer.jl` is a Julia package for automatic Bayesian inference on a factor graph with reactive message passing.

Given a probabilistic model, RxInfer allows for an efficient message-passing based Bayesian inference. It uses the model structure to generate an algorithm that consists of a sequence of local computations on a factor graph representation of the model.

> [!NOTE]
> RxInfer can also be used from Python through our [client-server infrastructure](#client-server-infrastructure) and [Python SDK](https://github.com/lazydynamics/RxInferClient.py).

### Performance and scalability

RxInfer.jl has been designed with a focus on efficiency, scalability and maximum performance for running Bayesian inference with message passing. Below is a comparison between RxInfer.jl and Turing.jl on latent state estimation in a linear multi-variate Gaussian state-space model. [Turing.jl](https://github.com/TuringLang/Turing.jl) is a state-of-the-art Julia-based general-purpose probabilistic programming package and is capable of running inference in a broader class of models. Still, RxInfer.jl executes the inference task in [various models](https://examples.rxinfer.com) faster and more accurately. RxInfer.jl accomplishes this by taking advantage of any conjugate likelihood-prior pairings in the model, which have analytical posteriors that are known by RxInfer.jl. As a result, in models with conjugate pairings, RxInfer.jl often beats general-purpose probabilistic programming packages in terms of computational load, speed, memory and accuracy. Note, however, that RxInfer.jl also supports non-conjugate inference and is continually improving in order to support a larger class of models.

Turing comparison             |  Scalability performance
:-------------------------:|:-------------------------:
![](benchmarks/plots/lgssm_comparison.svg?raw=true&sanitize=true)  |  ![](benchmarks/plots/lgssm_scaling.svg?raw=true&sanitize=true)

> [!NOTE]
> See many more examples in the [RxInferExamples.jl](https://examples.rxinfer.com/) repository.

### Faster inference with better results

RxInfer.jl not only beats generic-purpose Bayesian inference methods in conjugate models, executes faster, and scales better, but also provides more accurate results. Check out more examples [here](https://examples.rxinfer.com)!

Inference with RxInfer             |  Inference with HMC
:-------------------------:|:-------------------------:
![](benchmarks/plots/inference_rxinfer.svg?raw=true&sanitize=true)  |  ![](benchmarks/plots/inference_turing.svg?raw=true&sanitize=true)

The benchmark and accuracy experiment, which generated these plots, is available in the `benchmarks/` folder. Note, that the execution speed and accuracy 
of the HMC estimator heavily depends on the choice of hyperparameters. 
In this example, RxInfer executes exact inference consistently and does not depend on any hyperparameters.

### References

- [RxInfer: A Julia package for reactive real-time Bayesian inference](https://doi.org/10.21105/joss.05161) - a reference paper for the `RxInfer.jl` framwork.
- [Reactive Probabilistic Programming for Scalable Bayesian Inference](https://pure.tue.nl/ws/portalfiles/portal/313860204/20231219_Bagaev_hf.pdf) - a PhD dissertation outlining core ideas and principles behind `RxInfer` ([link2](https://research.tue.nl/nl/publications/reactive-probabilistic-programming-for-scalable-bayesian-inferenc), [link3](https://github.com/bvdmitri/phdthesis)).
- [Variational Message Passing and Local Constraint Manipulation in Factor Graphs](https://doi.org/10.3390/e23070807) - describes theoretical aspects of the underlying Bayesian inference method.
- [Reactive Message Passing for Scalable Bayesian Inference](https://doi.org/10.48550/arXiv.2112.13251) - describes implementation aspects of the Bayesian inference engine and performs benchmarks and accuracy comparison on various models.
- [A Julia package for reactive variational Bayesian inference](https://doi.org/10.1016/j.simpa.2022.100299) - a reference paper for the `ReactiveMP.jl` package, the underlying inference engine.

# Installation

Install RxInfer through the Julia package manager:

```
] add RxInfer
```

Optionally, use `] test RxInfer` to validate the installation by running the test suite.

# Documentation

For more information about `RxInfer.jl` please refer to the [documentation](https://docs.rxinfer.com).

# Getting Started

> [!NOTE]
> There are examples available to get you started in the [RxInferExamples.jl](https://examples.rxinfer.com/) repository.

### Coin flip simulation

Here we show a simple example of how to use RxInfer.jl for Bayesian inference problems. In this example we want to estimate a bias of a coin in a form of a probability distribution in a coin flip simulation.

First let's setup our environment by importing all needed packages:
```julia
using RxInfer, Random
```

We start by creating some dataset. For simplicity in this example we will use static pre-generated dataset. Each sample can be thought of as the outcome of single flip which is either heads or tails (1 or 0). We will assume that our virtual coin is biased, and lands heads up on 75% of the trials (on average).

```julia
n = 500  # Number of coin flips
p = 0.75 # Bias of a coin

distribution = Bernoulli(p) 
dataset      = rand(distribution, n)
```

### Model specification

In a Bayesian setting, the next step is to specify our probabilistic model. This amounts to specifying the joint probability of the random variables of the system.

#### Likelihood
We will assume that the outcome of each coin flip is governed by the Bernoulli distribution, i.e.

```math
y_i \sim \mathrm{Bernoulli}(\theta)
```

where $y_i = 1$ represents "heads", $y_i = 0$ represents "tails". The underlying probability of the coin landing heads up for a single coin flip is $\theta \in [0,1]$.

#### Prior
We will choose the conjugate prior of the Bernoulli likelihood function defined above, namely the beta distribution, i.e.

```math
\theta \sim Beta(a, b)
```

where $a$ and $b$ are the hyperparameters that encode our prior beliefs about the possible values of $\theta$. We will assign values to the hyperparameters in a later step.   

#### Joint probability
The joint probability is given by the multiplication of the likelihood and the prior, i.e.

```math
P(y_{1:N}, \theta) = P(\theta) \prod_{i=1}^N P(y_i | \theta).
```

Now let's see how to specify this model using [GraphPPL's package](https://github.com/ReactiveBayes/GraphPPL.jl) syntax:
```julia
# GraphPPL.jl export `@model` macro for model specification
# It accepts a regular Julia function and builds a factor graph under the hood
@model function coin_model(y, a, b) 
    # We endow θ parameter of our model with some prior
    θ ~ Beta(a, b)
    # We assume that outcome of each coin flip 
    # is governed by the Bernoulli distribution
    for i in eachindex(y)
        y[i] ~ Bernoulli(θ)
    end  
end
```

In short, the `@model` macro converts a textual description of a probabilistic model into a corresponding [Factor Graph](https://en.wikipedia.org/wiki/Factor_graph) (FG). In the example above, the $\theta \sim \mathrm{Beta}(a, b)$ expression creates latent variable $θ$ and assigns it as an output of $\mathrm{Beta}$ node in the corresponding factor graph. The `~` operation can be understood as _"is modelled by"_. Next, we model each data point `y[i]` as $\mathrm{Bernoulli}$ distribution with $\theta$ as its parameter.

> [!TIP]
> Alternatively, we could use the broadcasting operation:
```julia
@model function coin_model(y, a, b) 
    θ  ~ Beta(a, b)
    y .~ Bernoulli(θ) 
end
```

As you can see, `RxInfer` in combination with `GraphPPL` offers a model specification syntax that resembles closely to the mathematical equations defined above. 

### Inference specification

Once we have defined our model, the next step is to use `RxInfer` API to infer quantities of interests. To do this we can use a generic `infer` function from `RxInfer.jl` that supports static datasets.

```julia
result = infer(
    model = coin_model(a = 2.0, b = 7.0),
    data  = (y = dataset, )
)
```

![Coin Flip](docs/src/assets/img/coin-flip.svg?raw=true&sanitize=true "Coin-Flip readme results")

# Where to go next?

There are a set of [examples](https://examples.rxinfer.com/) available in the `RxInferExamples.jl` repository that demonstrate the more advanced features of the package. Alternatively, you can head to the [documentation](https://docs.rxinfer.com) that provides more detailed information of how to use `RxInfer` to specify more complex probabilistic models.

# Roadmap

For a more granular view of our progress and ongoing tasks, check out our [project board](https://github.com/orgs/reactivebayes/projects/2/views/4) or join our 4-weekly [public meetings](https://dynalist.io/d/F4aA-Z2c8X-M1iWTn9hY_ndN).

## Ecosystem

The `RxInfer` framework consists of four *core* packages developed by [ReactiveBayes](https://github.com/reactivebayes/):

- [`ReactiveMP.jl`](https://github.com/reactivebayes/ReactiveMP.jl) - the underlying message passing-based inference engine
- [`GraphPPL.jl`](https://github.com/reactivebayes/GraphPPL.jl) - model and constraints specification package
- [`ExponentialFamily.jl`](https://github.com/reactivebayes/ExponentialFamily.jl) - package for exponential family distributions
- [`Rocket.jl`](https://github.com/reactivebayes/Rocket.jl) - reactive extensions package for Julia 

## NumFocus

We are very proud to be an affiliated project with [NumFocus](https://numfocus.org/), a non-profit organization that supports the open-source scientific computing community.

![](docs/src/assets/numfocus.png)

## Client-Server Infrastructure and Python integration

> [!NOTE]
> The development of RxInferServer has been sponsored by [Lazy Dynamics](https://lazydynamics.com/). The repositories are hosted under the [Lazy Dynamics](https://github.com/lazydynamics/) organization. Note that LICENSE for the RxInferServer is different from RxInfer.

RxInfer can be deployed as a RESTful API service using [`RxInferServer`](https://github.com/lazydynamics/RxInferServer), which provides:

- OpenAPI-compliant RESTful API endpoints for RxInfer models
- Support for model instance management and inference execution
- Real-time inference capabilities
- Comprehensive API documentation at [server.rxinfer.com](https://server.rxinfer.com)

### Client SDKs

To interact with RxInferServer, you can use one of the following SDKs:

- **Python SDK**: [`RxInferClient.py`](https://github.com/lazydynamics/RxInferClient.py) - A Python client for interacting with RxInferServer
- **TypeScript SDK**: [`RxInferClient.ts`](https://github.com/lazydynamics/RxInferClient.ts) - A TypeScript client for interacting with RxInferServer
- **Julia SDK**: Included in the RxInferServer repository

SDKs provide a convenient interface to:
- Create and manage model instances
- Execute inference tasks
- Monitor inference progress
- Handle authentication and API keys
- Process results in a native format

# Join Our Community and Contribute to RxInfer

RxInfer is a community-driven project and we welcome all contributions! To get started:
- Check out our [contributing guide](https://docs.rxinfer.com/stable/contributing/guide)
- Review the [contributing guidelines](https://docs.rxinfer.com/stable/contributing/guidelines)
- Browse [beginner-friendly issues](https://github.com/ReactiveBayes/RxInfer.jl/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) to find something that interests you

## Active Inference Institute Collaboration

The [Active Inference Institute](https://www.activeinference.org/) community members are enhancing RxInfer/GraphPPL's visualization capabilities. Their work includes:
- Developing advanced model visualization features ✅ ([PR](https://github.com/ReactiveBayes/GraphPPL.jl/pull/251))
- Creating summary and subgraph visualization modalities
- Implementing various graph layout algorithms
- Improving model inspection and understanding tools

For more details on ongoing work, see the [RxInfer development project board](https://coda.io/d/RxInfer-2024-Active-Inference-Institute_ddtS-XZ4BJb/Developing-RxInfer-jl_sufeCeIa#_lutfq_7F).

## Learning Resources

The community maintains educational content and tutorials on [Learnable Loop](https://learnableloop.com/#category=RxInfer), covering topics such as:
- Visualizing Forney Factor Graphs
- Sales forecasting with time-varying autoregressive models
- Hidden Markov models with control
- Applications of Active Inference across different domains

## JuliaCon 2023 presentation

Additionally, checkout our [video from JuliaCon 2023](https://www.youtube.com/watch?v=qXrvDVm_fnE) for a high-level overview of the package

<p align="center">
    <a href="https://www.youtube.com/watch?v=qXrvDVm_fnE"><img style="width: 100%" src="https://img.youtube.com/vi/qXrvDVm_fnE/0.jpg"></a>
</p>

## Our presentation at the Julia User Group Munich meetup

Also check out the recorded presentation at the Julia User Group Munich meetup for a more detailed overview of the package

<p align="center">
    <a href="https://www.youtube.com/watch?v=KuluqEzFtm8"><img style="width: 100%" src="https://img.youtube.com/vi/KuluqEzFtm8/0.jpg"></a>
</p>


# Telemetry

RxInfer collects completely anonymous telemetry data regarding package usage. This information helps us understand how RxInfer is used and shapes our roadmap to prioritize features and improvements. The telemetry:
- Does not collect any code, data, or environment information, only the fact of `using RxInfer` once per Julia session
- Entirely anonymous
- (Opt-out) Can be disabled for a single Julia session or permanently

You can learn more about it and how to opt-out by visiting our [documentation](https://docs.rxinfer.com).

## Session Sharing

RxInfer includes an optional session sharing feature that can help us provide better support and improve the package. When you encounter an issue, you can share your session data with us, which includes:
- Model source code and metadata
- Input data characteristics (no actual data)
- Execution timing and success rates
- Error information (if any)
- Environment information (Julia version, OS, etc.)

This information is invaluable for debugging issues and improving RxInfer. Session sharing is:
- Completely optional and disabled by default
- Entirely anonymous
- Only shared when you explicitly choose to do so
- (Opt-in) Can be enabled to send reports automatically when an error occurs. When enabled, still entirely anonymous.

If you're opening a GitHub issue, we encourage you to share your session ID with us - it helps us understand your use case better and provide more accurate support. Learn more about session sharing and how to opt-in in our [documentation](https://docs.rxinfer.com).

# License

[MIT License](LICENSE) Copyright (c) 2021-2024 BIASlab, 2024-present ReactiveBayes
