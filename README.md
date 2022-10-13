
<p align="center">
  <img src="https://user-images.githubusercontent.com/6557701/173435965-2020adae-b10b-41d2-9c7f-98ad63986958.svg" width=50%/>
</p>

---

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://biaslab.github.io/RxInfer.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://biaslab.github.io/RxInfer.jl/dev/)
[![Build Status](https://github.com/biaslab/RxInfer.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/biaslab/RxInfer.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/biaslab/RxInfer.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/biaslab/RxInfer.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/R/RxInfer.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/report.html)

RxInfer.jl is a Julia package for automatic Bayesian inference on a factor graph with reactive message passing.

Given a probabilistic model, RxInfer allows for an efficient message-passing based Bayesian inference. It uses the model structure to generate an algorithm that consists of a sequence of local computations on a Forney-style factor graph (FFG) representation of the model.

The current version supports belief propagation (sum-product message passing) and variational message passing (both Mean-Field and Structured VMP)

RxInfer.jl has been designed with a focus on efficiency, scalability and maximum performance for running inference on conjugate state-space models with message passing. Below is a benchmark comparison between RxInfer.jl and [Turing.jl](https://github.com/TuringLang/Turing.jl) on a linear multivariate Gaussian state space model. RxInfer.jl does execute the inference procedure faster and also provides more accurate analytical solution to the problem. This model contains many conjugate prior and likelihood pairings that lead to analytically computable Bayesian posteriors. For these types of models, RxInfer.jl takes advantage of the conjugate pairings and beats general-purpose probabilistic programming packages easily in terms of computational load, speed, memory and accuracy.

Turing comparison             |  Scalability performance
:-------------------------:|:-------------------------:
![](benchmarks/plots/lgssm_comparison.svg?raw=true&sanitize=true)  |  ![](benchmarks/plots/lgssm_scaling.svg?raw=true&sanitize=true)

# Installation

Install RxInfer through the Julia package manager:

```
] add RxInfer
```

Optionally, use `] test RxInfer` to validate the installation by running the test suite.

# Getting Started

There are examples available to get you started in the `examples/` folder. 

### Coin flip simulation

Here we show a simple example of how to use RxInfer.jl for Bayesian inference problems. In this example we want to estimate a bias of a coin in a form of a probability distribution in a coin flip simulation.

Let's start by creating some dataset. For simplicity in this example we will use static pre-generated dataset. Each sample can be thought of as the outcome of single flip which is either heads or tails (1 or 0). We will assume that our virtual coin is biased, and lands heads up on 75% of the trials (on average).

First let's setup our environment by importing all needed packages:

```julia
using RxInfer, Distributions, Random
```

Next, let's define our dataset:

```julia
n = 500  # Number of coin flips
p = 0.75 # Bias of a coin

distribution = Bernoulli(p) 
dataset      = float.(rand(Bernoulli(p), n))
```

### Model specification

In a Bayesian setting, the next step is to specify our probabilistic model. This amounts to specifying the joint probability of the random variables of the system.

#### Likelihood
We will assume that the outcome of each coin flip is governed by the Bernoulli distribution, i.e.

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=y_i%20\sim%20\mathrm{Bernoulli}(\theta)">
</p>

where <img src="https://render.githubusercontent.com/render/math?math=y_1%20=%201"> represents "heads", <img src="https://render.githubusercontent.com/render/math?math=y_1%20=%200"> represents "tails". The underlying probability of the coin landing heads up for a single coin flip is <img src="https://render.githubusercontent.com/render/math?math=\theta%20\in%20[0,1]">.

#### Prior
We will choose the conjugate prior of the Bernoulli likelihood function defined above, namely the beta distribution, i.e.

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=\theta%20\sim%20Beta(a,%20b)">
</p>

where ``a`` and ``b`` are the hyperparameters that encode our prior beliefs about the possible values of ``θ``. We will assign values to the hyperparameters in a later step.   

#### Joint probability
The joint probability is given by the multiplication of the likelihood and the prior, i.e.

<p align="center">
    <img src="https://render.githubusercontent.com/render/math?math=P(y_{1:N},%20\theta)%20=%20P(\theta)%20\prod_{i=1}^N%20P(y_i%20|%20\theta).">
</p>

Now let's see how to specify this model using GraphPPL's package syntax.

```julia

# GraphPPL.jl export `@model` macro for model specification
# It accepts a regular Julia function and builds an FFG under the hood
@model function coin_model(n)

    # `datavar` creates data 'inputs' in our model
    # We will pass data later on to these inputs
    # In this example we create a sequence of inputs that accepts Float64
    y = datavar(Float64, n)
    
    # We endow θ parameter of our model with some prior
    θ ~ Beta(2.0, 7.0)
    
    # We assume that outcome of each coin flip 
    # is governed by the Bernoulli distribution
    for i in 1:n
        y[i] ~ Bernoulli(θ)
    end
    
    # We return references to our data inputs and θ parameter
    # We will use these references later on during inference step
    return y, θ
end

```

As you can see, `RxInfer` offers a model specification syntax that resembles closely to the mathematical equations defined above. We use `datavar` function to create "clamped" variables that take specific values at a later date. `θ ~ Beta(2.0, 7.0)` expression creates random variable `θ` and assigns it as an output of `Beta` node in the corresponding FFG. 

### Inference specification

Once we have defined our model, the next step is to use `RxInfer` API to infer quantities of interests. To do this we can use a generic `inference` function from `RxInfer.jl` that supports static datasets.

```julia
result = inference(
    model = coin_model(length(dataset)),
    data  = (y = dataset, )
)
```

![Coin Flip](docs/src/assets/img/coin-flip.svg?raw=true&sanitize=true "Coin-Flip readme results")

# Where to go next?
There are a set of [examples](https://github.com/biaslab/ReactiveMP.jl/tree/master/examples) available in `RxInfer` repository that demonstrate the more advanced features of the package. Alternatively, you can head to the [documentation][docs-stable-url] that provides more detailed information of how to use `RxInfer` to specify more complex probabilistic models.

See also:
- [`ReactiveMP.jl`](https://github.com/biaslab/ReactiveMP.jl)
- [`GraphPPL.jl`](https://github.com/biaslab/GraphPPL.jl)
- [`Rocket.jl`](https://github.com/biaslab/Rocket.jl)

# License

MIT License Copyright (c) 2021-2022 BIASlab
