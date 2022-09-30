# [Getting started](@id user-guide-getting-started)

`RxInfer.jl` is a Julia package for Bayesian Inference on Factor Graphs by Message Passing. It supports both exact and variational inference algorithms.

`RxInfer.jl` package forms an ecosystem around three main packages: `ReactiveMP.jl` exports a reactive message passing based Bayesian inference engine, `Rocket.jl` is the core library that enables reactivity and `GraphPPL.jl` library simplifies model and constraints specification. `ReactiveMP.jl` engine is a successor of the [`ForneyLab`](https://github.com/biaslab/ForneyLab.jl) package. It follows the same ideas and concepts for message-passing based inference, but uses new reactive and efficient message passing implementation under the hood. The API between two packages is different due to a better flexibility, performance and new reactive approach for solving inference problems.

This page provides the necessary information you need to get started with `Rxinfer`. We will show the general approach to solving inference problems with `RxInfer` by means of a running example: inferring the bias of a coin.

## Installation

Install `RxInfer` through the Julia package manager:
```
] add RxInfer
```

## Importing ReactiveMP

To add `RxInfer` package (and all associated packages) into a running Julia session simply run:

```julia
using RxInfer
```

Read more about about `using` in the [Using methods from RxInfer](@ref lib-using-methods) section of the documentation.

## Example: Inferring the bias of a coin
The `RxInfer` approach to solving inference problems consists of three phases:

1. [Model specification](@ref getting-started-model-specification): `RxInfer` uses `GraphPPL` package for model specification part. It offers a domain-specific language to specify your probabilistic model.
2. [Inference specification](@ref getting-started-inference-specification): `RxInfer` inference API uses `ReactiveMP` inference engine under the hood and has been designed to be as flexible as possible. It is compatible both with asynchronous infinite data streams and with static datasets. For most of the use cases it consists of the same simple building blocks. In this example we will show one of the many possible ways to infer your quantities of interest.
3. [Inference execution](@ref getting-started-inference-execution): Given model specification and inference procedure it is pretty straightforward to use reactive API from `Rocket` to pass data to the inference backend and to run actual inference.

### Coin flip simulation
Let's start by creating some dataset. One approach could be flipping a coin N times and recording each outcome. For simplicity in this example we will use static pre-generated dataset. Each sample can be thought of as the outcome of single flip which is either heads or tails (1 or 0). We will assume that our virtual coin is biased, and lands heads up on 75% of the trials (on average).

First let's setup our environment by importing all needed packages:

```@example coin
using RxInfer, Distributions, Random
```

Next, let's define our dataset:

```@example coin
rng = MersenneTwister(42)
n = 10
p = 0.75
distribution = Bernoulli(p)

dataset = float.(rand(rng, Bernoulli(p), n))
```

### [Model specification](@id getting-started-model-specification)

In a Bayesian setting, the next step is to specify our probabilistic model. This amounts to specifying the joint probability of the random variables of the system.

#### Likelihood
We will assume that the outcome of each coin flip is governed by the Bernoulli distribution, i.e.

```math 
y_i \sim \mathrm{Bernoulli}(\theta),
```

where ``y_i = 1`` represents "heads", ``y_i = 0`` represents "tails". The underlying probability of the coin landing heads up for a single coin flip is ``\theta \in [0,1]``.

#### Prior
We will choose the conjugate prior of the Bernoulli likelihood function defined above, namely the beta distribution, i.e.

```math 
\theta \sim Beta(a, b),
```

where ``a`` and ``b`` are the hyperparameters that encode our prior beliefs about the possible values of ``\theta``. We will assign values to the hyperparameters in a later step.   

#### Joint probability
The joint probability is given by the multiplication of the likelihood and the prior, i.e.

```math
P(y_{1:N}, θ) = P(θ) \prod_{i=1}^N P(y_i | θ).
```

Now let's see how to specify this model using GraphPPL's package syntax.

```@example coin

# GraphPPL.jl export `@model` macro for model specification
# It accepts a regular Julia function and builds an FFG under the hood
@model function coin_model(n)

    # `datavar` creates data 'inputs' in our model
    # We will pass data later on to these inputs
    # In this example we create a sequence of inputs that accepts Float64
    y = datavar(Float64, n)
    
    # We endow θ parameter of our model with some prior
    θ ~ Beta(2.0, 7.0)
    # or, in this particular case, the `Uniform(0.0, 1.0)` prior also works:
    # θ ~ Uniform(0.0, 1.0)
    
    # We assume that outcome of each coin flip is governed by the Bernoulli distribution
    for i in 1:n
        y[i] ~ Bernoulli(θ)
    end
    
    # We return references to our data inputs and θ parameter
    # We will use these references later on during inference step
    return y, θ
end

```

As you can see, `RxInfer` offers a model specification syntax that resembles closely to the mathematical equations defined above. We use `datavar` function to create "clamped" variables that take specific values at a later date. `θ ~ Beta(2.0, 7.0)` expression creates random variable `θ` and assigns it as an output of `Beta` node in the corresponding FFG. 

!!! note
    To quickly check the list of all available factor nodes that can be used in the model specification language call `?make_node` or `Base.doc(make_node)`.

### [Inference specification](@id getting-started-inference-specification)

#### Automatic inference specification

Once we have defined our model, the next step is to use `RxInfer` API to infer quantities of interests. To do this we can use a generic `inference` function that supports static datasets.

```@example coin 
result = inference(
    model = coin_model(length(dataset)),
    data  = (y = dataset, )
)
```

```@example coin 
θestimated = result.posteriors[:θ]
```

```@example coin
println("mean: ", mean(θestimated))
println("std:  ", std(θestimated))
nothing #hide
```

Read more information about the `inference` function in the [Inference execution](@ref user-guide-inference-execution-automatic-specification) documentation section.

#### Manual inference specification

There is a way to manually specify an inference procedure for advanced use-cases. `RxInfer` API is flexible in terms of inference specification and is compatible both with real-time inference processing and with static datasets. In most of the cases for static datasets, as in our example, it consists of same basic building blocks:

1. Return variables of interests from model specification
2. Subscribe on variables of interests posterior marginal updates
3. Pass data to the model
4. Unsubscribe 

Here is an example of inference procedure:

```@example coin 
function custom_inference(data)
    n = length(data)

    # `coin_model` function from `@model` macro returns a reference to the model generator object
    # we need to use the `create_model` function to get actual model object
    model, (y, θ) = create_model(coin_model(n))
    
    # Reference for future posterior marginal 
    mθ = nothing

    # `getmarginal` function returns an observable of future posterior marginal updates
    # We use `Rocket.jl` API to subscribe on this observable
    # As soon as posterior marginal update is available we just save it in `mθ`
    subscription = subscribe!(getmarginal(θ), (m) -> mθ = m)
    
    # `update!` function passes data to our data inputs
    update!(y, data)
    
    # It is always a good practice to unsubscribe and to 
    # free computer resources held by the subscription
    unsubscribe!(subscription)
    
    # Here we return our resulting posterior marginal
    return mθ
end
```

### [Inference execution](@id getting-started-inference-execution)

Here after everything is ready we just call our `inference` function to get a posterior marginal distribution over `θ` parameter in the model.

```@example coin
θestimated = custom_inference(dataset)
```

```@example coin
println("mean: ", mean(θestimated))
println("std:  ", std(θestimated))
nothing #hide
```

```@example coin
using Plots, LaTeXStrings; theme(:default)

rθ = range(0, 1, length = 1000)

p1 = plot(rθ, (x) -> pdf(Beta(2.0, 7.0), x), title="Prior", fillalpha=0.3, fillrange = 0, label=L"P\:(\theta)", c=1,)
p2 = plot(rθ, (x) -> pdf(θestimated, x), title="Posterior", fillalpha=0.3, fillrange = 0, label=L"P\:(\theta|y)", c=3)

plot(p1, p2, layout = @layout([ a; b ]))
```

In our dataset we used 10 coin flips to estimate the bias of a coin. It resulted in a vague posterior distribution, however `ReactiveMP` scales very well for large models and factor graphs. We may use more coin flips in our dataset for better posterior distribution estimates:

```@example coin
dataset_100   = float.(rand(rng, Bernoulli(p), 100))
dataset_1000  = float.(rand(rng, Bernoulli(p), 1000))
dataset_10000 = float.(rand(rng, Bernoulli(p), 10000))
nothing # hide
```

```@example coin
θestimated_100   = custom_inference(dataset_100)
θestimated_1000  = custom_inference(dataset_1000)
θestimated_10000 = custom_inference(dataset_10000)
nothing #hide
```

```@example coin
p3 = plot(title = "Posterior", legend = :topleft)

p3 = plot!(p3, rθ, (x) -> pdf(θestimated_100, x), fillalpha = 0.3, fillrange = 0, label = "P(θ|y_100)", c = 4)
p3 = plot!(p3, rθ, (x) -> pdf(θestimated_1000, x), fillalpha = 0.3, fillrange = 0, label = "P(θ|y_1000)", c = 5)
p3 = plot!(p3, rθ, (x) -> pdf(θestimated_10000, x), fillalpha = 0.3, fillrange = 0, label = "P(θ|y_10000)", c = 6)

plot(p1, p3, layout = @layout([ a; b ]))
```

With larger dataset our posterior marginal estimate becomes more and more accurate and represents real value of the bias of a coin.

```@example coin
println("mean: ", mean(θestimated_10000))
println("std:  ", std(θestimated_10000))
nothing #hide
```

## Where to go next?
There are a set of [examples](https://github.com/biaslab/ReactiveMP.jl/tree/master/examples) available in `RxInfer` repository that demonstrate the more advanced features of the package and also [Examples](@ref examples-overview) section in the documentation. Alternatively, you can head to the [Model specification](@ref user-guide-model-specification) which provides more detailed information of how to use `RxInfer` to specify probabilistic models. [Inference execution](@ref user-guide-inference-execution) section provides a documentation about `RxInfer` API for running reactive Bayesian inference.
