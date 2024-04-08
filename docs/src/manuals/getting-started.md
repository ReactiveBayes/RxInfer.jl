# [Getting started](@id user-guide-getting-started)

`RxInfer.jl` is a Julia package for Bayesian Inference on Factor Graphs by Message Passing. 
It supports both exact and variational inference algorithms and forms an ecosystem around three main packages: 
- [`ReactiveMP.jl`](https://github.com/reactivebayes/ReactiveMP.jl) - the underlying message passing-based inference engine
- [`GraphPPL.jl`](https://github.com/reactivebayes/GraphPPL.jl) - model and constraints specification package
- [`Rocket.jl`](https://github.com/reactivebayes/Rocket.jl) - reactive extensions package for Julia 

This page provides the necessary information you need to get started with `Rxinfer`. We will show the general approach to solving inference problems with `RxInfer` by means of a running example: inferring the bias of a coin using a simple Beta-Bernoulli model.

## Installation

`RxInfer` is an officially registered Julia package. Install `RxInfer` through the Julia package manager by using the following command from the package manager mode:

```julia
] add RxInfer
```

Alternatively:

```
julia> import Pkg

julia> Pkg.add("RxInfer")
```

## Importing RxInfer

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

### [Coin flip simulation](@id user-guide-getting-started-coin-flip-simulation)

Let's start by creating some dataset. One approach could be flipping a coin N times and recording each outcome. For simplicity in this example we will use static pre-generated dataset. Each sample can be thought of as the outcome of single flip which is either heads or tails (1 or 0). We will assume that our virtual coin is biased, and lands heads up on 75% of the trials (on average).

First let's setup our environment by importing all needed packages:

```@example coin
using RxInfer, Distributions, Random
```

Next, let's define our dataset:

```@example coin
# Random number generator for reproducibility
rng            = MersenneTwister(42)
# Number of coin flips (observations)
n_observations = 10
# The bias of a coin used in the demonstration
coin_bias      = 0.75
# We assume that the outcome of each coin flip is 
# distributed as the `Bernoulli` distrinution
distribution   = Bernoulli(coin_bias)
# Simulated coin flips
dataset        = float.(rand(rng, distribution, n_observations))
```

### [Model specification](@id getting-started-model-specification)

In a Bayesian setting, the next step is to specify our probabilistic model. This amounts to specifying the joint probability of the random variables of the system.

#### Likelihood

We've made an assumption that the outcome of each coin flip is governed by the [`Bernoulli`](https://en.wikipedia.org/wiki/Bernoulli_distribution) distribution, i.e.

```math 
y_i \sim \mathrm{Bernoulli}(\theta),
```

where ``y_i = 1`` represents "heads", ``y_i = 0`` represents "tails". The underlying probability of the coin landing heads up for a single coin flip is ``\theta \in [0,1]``.

#### Prior

We will choose the conjugate prior of the `Bernoulli` likelihood function defined above, namely the [`Beta`](https://en.wikipedia.org/wiki/Beta_distribution) distribution, i.e.

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
@model function coin_model(y, a, b)
    # We endow θ parameter of our model with some prior
    θ ~ Beta(a, b)
    # or, in this particular case, the `Uniform(0.0, 1.0)` prior also works:
    # θ ~ Uniform(0.0, 1.0)

    # We assume that outcome of each coin flip is governed by the Bernoulli distribution
    for i in eachindex(y)
        y[i] ~ Bernoulli(θ)
    end
end
```

As you can see, `RxInfer` offers a model specification syntax that resembles closely to the mathematical equations defined above.
Alternatively, we could use a broadcasting syntax:

```@example coin
@model function coin_model(y, a, b) 
    θ  ~ Beta(a, b)
    y .~ Bernoulli(θ) 
end
```

!!! note
    To quickly check the list of all available factor nodes that can be used in the model specification language call `?ReactiveMP.is_predefined_node` or `Base.doc(ReactiveMP.is_predefined_node)`.

### [Conditioning on data and inspecting the model structure](@id getting-started-conditioning)

Given the model specification we can construct an actual model graph and visualize it. In order to do that we can use the `|` operator to condition on data and the `RxInfer.create_model` function to create the graph. Read more about condition in the [corresponding section](@ref user-guide-model-specification-conditioning) of the documentation.

```@example coin
conditioned = coin_model(a = 2.0, b = 7.0) | (y = [ 1.0, 0.0, 1.0 ], )
```

We can use `GraphPPL.jl` visualisation capabilities to show the structure of the resulting graph. For that we need two extra packages installed: `Cairo` and `GraphPlot`. Note, that those packages are not included in the `RxInfer` package and must be installed separately.

```@example coin
using Cairo, GraphPlot

# `Create` the actual graph of the model conditioned on the data
model = RxInfer.create_model(conditioned)

# Call `gplot` function from `GraphPlot` to visualise the structure of the graph
GraphPlot.gplot(RxInfer.getmodel(model))
```

In addition, we can also programatically query the structure of the graph:

```@example coin
RxInfer.getrandomvars(model)
```

```@example coin
RxInfer.getdatavars(model)
```

```@example coin
RxInfer.getconstantvars(model)
```

```@example coin
RxInfer.getfactornodes(model)
```

### Conditioning on data that is not available at model creation time

Sometimes the data is not known at model creation time, for example, during reactive inference.
For that purpose `RxInfer` uses [`RxInfer.DefferedDataHandler`](@ref) structure.

```@example coin
# The only difference here is that we do not specify `a` and `b` as hyper-parameters 
# But rather indicate that the data for them will be available later during the inference
conditioned_with_deffered_data = coin_model() | (
    y = [ 1.0, 0.0, 1.0 ], 
    a = RxInfer.DefferedDataHandler(), 
    b = RxInfer.DefferedDataHandler()
)

# The graph creation API does not change
model_with_deffered_data = RxInfer.create_model(conditioned_with_deffered_data)

# We can visualise the graph with missing data handles as well
GraphPlot.gplot(RxInfer.getmodel(model_with_deffered_data))
```

From the model structure visualisation we can see now that both `a` and `b` are no longer indicated as constants.
Read more about the structure of the graph in [`GraphPPL` documentation](https://reactivebayes.github.io/GraphPPL.jl/stable/).

### [Inference specification](@id getting-started-inference-specification)

#### Automatic inference specification

Once we have defined our model, the next step is to use `RxInfer` API to infer quantities of interests. To do this we can use a generic [`infer`](@ref) function that supports static datasets.
Read more information about the [`infer`](@ref) function in the [Inference Execution](@ref user-guide-inference-execution) documentation section.

```@example coin 
result = infer(
    model = coin_model(a = 2.0, b = 7.0),
    data  = (y = dataset, )
)
```

As you can see we don't need to condition on the data manually, the [`infer`](@ref) function will do it automatically.
After the inference is complete we can fetch the results from the `.posterior` field with the name of the latent state:

```@example coin 
θestimated = result.posteriors[:θ]
```

We can also compute some statistical properties of the result:

```@example coin
println("Real bias is ", coin_bias)
println("Estimated bias is ", mean(θestimated))
println("Standard deviation ", std(θestimated))
nothing #hide
```

Let's also visualize the resulting posteriors:

```@example coin
using Plots

rθ = range(0, 1, length = 1000)

p1 = plot(rθ, (x) -> pdf(Beta(2.0, 7.0), x), title="Prior", fillalpha=0.3, fillrange = 0, label="P(θ)", c=1,)
p2 = plot(rθ, (x) -> pdf(θestimated, x), title="Posterior", fillalpha=0.3, fillrange = 0, label="P(θ|y)", c=3)

plot(p1, p2, layout = @layout([ a; b ]))
```

In our dataset we used 10 coin flips and skewed prior to estimate the bias of a coin. 
It resulted in a vague posterior distribution, however `RxInfer` scales very well for large models and factor graphs. 
We may use more coin flips in our dataset for better posterior distribution estimates:

```@example coin
dataset_100   = float.(rand(rng, Bernoulli(coin_bias), 100))
dataset_1000  = float.(rand(rng, Bernoulli(coin_bias), 1000))
dataset_10000 = float.(rand(rng, Bernoulli(coin_bias), 10000))
nothing # hide
```

```@example coin
θestimated_100   = infer(model = coin_model(a = 2.0, b = 7.0), data  = (y = dataset_100, ))
θestimated_1000  = infer(model = coin_model(a = 2.0, b = 7.0), data  = (y = dataset_1000, ))
θestimated_10000 = infer(model = coin_model(a = 2.0, b = 7.0), data  = (y = dataset_10000, ))
nothing #hide
```

Let's investigate how the number of observation affects the estimated posterior:

```@example coin
p3 = plot(title = "Posterior", legend = :topleft)
p3 = plot!(p3, rθ, (x) -> pdf(θestimated_100.posteriors[:θ], x), fillalpha = 0.3, fillrange = 0, label = "P(θ|y_100)", c = 4)
p3 = plot!(p3, rθ, (x) -> pdf(θestimated_1000.posteriors[:θ], x), fillalpha = 0.3, fillrange = 0, label = "P(θ|y_1000)", c = 5)
p3 = plot!(p3, rθ, (x) -> pdf(θestimated_10000.posteriors[:θ], x), fillalpha = 0.3, fillrange = 0, label = "P(θ|y_10000)", c = 6)

plot(p1, p3, layout = @layout([ a; b ]))
```

We can see that with larger dataset our posterior marginal estimate becomes more and more accurate and represents real value of the bias of a coin.

```@example coin
println("Real bias is ", coin_bias)
println("Estimated bias is ", mean(θestimated_10000.posteriors[:θ]))
println("Standard deviation ", std(θestimated_10000.posteriors[:θ]))
nothing #hide
```

## Where to go next?

There are a set of [examples](@ref examples-overview) available in `RxInfer` repository that demonstrate the more advanced features of the package for various problems. Alternatively, you can head to the [Model specification](@ref user-guide-model-specification) which provides more detailed information of how to use `RxInfer` to specify probabilistic models. [Inference execution](@ref user-guide-inference-execution) section provides a documentation about `RxInfer` API for running reactive Bayesian inference. Also read the [Comparison](@ref comparison) to compare `RxInfer` with other probabilistic programming libraries.
