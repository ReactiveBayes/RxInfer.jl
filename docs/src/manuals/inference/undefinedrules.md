# [Inference without explicit message update rules](@id inference-undefinedrules)

`RxInfer` utilizes the [`ReactiveMP.jl`](https://github.com/ReactiveBayes/ReactiveMP.jl) package as its inference backend. Typically, running inference with `ReactiveMP.jl` requires users to define a factor node using the `@node` macro and specify corresponding message update rules with the `@rule` macro. For background on what rules are and how they work, see [Understanding Rules](@ref what-is-a-rule). Detailed instructions on implementing rules can be found in [this section](@ref create-node) of the documentation. However, in this tutorial, we will explore an alternative approach that allows inference with default message update rule for custom factor nodes by defining only `BayesBase.logpdf` and `BayesBase.insupport` for a factor node, without needing explicit `@rule` specifications.

!!! note 
    In the context of message-passing based Bayesian inference, custom message update rules enhance precision and efficiency. These rules leverage the specific mathematical properties of the model's distributions and relationships, leading to more accurate updates and faster convergence. By incorporating domain-specific knowledge, custom rules improve the robustness and reliability of the inference process, particularly in complex models where default rules may be inadequate or inefficient.

## A simple prior-likelihood model

We start a simple model with a hidden variable `p` and observations `y`. Later in the tutorial we explore more advanced use-cases.
In this particular case we assume that `p` follows a prior distribution and `y` are drawn from a likelihood distribution. The model can be defined as follows:

```@example inference-undefinedrules
using RxInfer

@model function simple_model(y, prior, likelihood)
    p ~ prior
    y .~ likelihood(p)
end
```

## Node specifications

Next, we define structures for both the `prior` and the `likelihood`. Let's start with the prior. Assume that the `p` parameter is best described by a `Beta` distribution. We can define it as follows:

!!! note
    The `Distributions.jl` package already provides a fully-featured implementation of `Beta` and `Bernoulli` distributions, including functions like `logpdf` and support checks. The example below redefines the `Beta` distribution structure and related functions solely for illustrative purposes. In practice, you often won't need to define these distributions yourself, as many of them has already been included in `Distributions.jl`.

```@example inference-undefinedrules
using Distributions, BayesBase

struct BetaDistribution{A, B} <: ContinuousUnivariateDistribution
    a::A
    b::B
end

# Reuse `logpdf` from `Distributions.jl` for illustrative purposes
BayesBase.logpdf(d::BetaDistribution, x) = logpdf(Beta(d.a, d.b), x)
BayesBase.insupport(d::BetaDistribution, x::Real) = 0 <= x <= 1
```

Next, we assume that `y` is a discrete dataset of `true` and `false` values. The logical choice for the `likelihood` distribution is the `Bernoulli` distribution.

```@example inference-undefinedrules
struct BernoulliDistribution{P} <: DiscreteUnivariateDistribution
    p::P
end

# Reuse `logpdf` from `Distributions.jl` for illustrative purposes
BayesBase.logpdf(d::BernoulliDistribution, x) = logpdf(Bernoulli(d.p), x)
BayesBase.insupport(d::BernoulliDistribution, x) = x === true || x === false
```

The next step is to register these structures as valid factor nodes:

```@example inference-undefinedrules
@node BetaDistribution Stochastic [out, a, b]
@node BernoulliDistribution Stochastic [out, p]
```

When specifying a node for our custom distributions, we must follow a specific edge ordering. The first edge is always `out`, which represents a sample in the `logpdf` function. All remaining edges must match the parameters of the distribution in the exact same order. For example, for the `BetaDistribution`, the node function is defined as `(out, a, b) -> logpdf(BetaDistribution(a, b), out)`. This ensures that the node specification and the `logpdf` function correctly maps the distribution parameters to the sample output.

!!! note
    Although `Beta` is a conjugate prior for the parameter of the `Bernoulli` distribution, `ReactiveMP` and `RxInfer` are unaware of this and cannot exploit this information. To utilize conjugacy, refer to the [custom node creation section](@ref create-node) of the documentation.

## Generating a synthetic dataset

Previously, we assumed that our dataset consists of discrete values: `true` and `false`. We can generate a synthetic dataset with these values as follows:

```@example inference-undefinedrules
using StableRNGs, Plots

hidden_p    = 1 / 3.1415 # a value between `0` and `1`
ndatapoints = 1_000      # number of observarions
dataset     = rand(StableRNG(42), Bernoulli(hidden_p), ndatapoints)

bar(["true", "false"], [ count(==(true), dataset), count(==(false), dataset) ], label = "dataset")
```

## Inference with a rule fallback

Now, we can run inference with `RxInfer`. Since explicit rules for our nodes have not defined, we can instruct the `ReactiveMP` backend to use fallback message update rules. Refer to the `ReactiveMP` documentation for available fallbacks. In this example, we will use the `NodeFunctionRuleFallback` structure, which uses the `logpdf` of the stochastic node to approximate messages.

!!! note
    `NodeFunctionRuleFallback` employs a simple approximation for outbound messages, which may significantly degrade inference accuracy. Whenever possible, it is recommended to define [proper message update rules](@ref create-node).

To complete the inference setup, we must define an approximation method for posteriors using the `@constraints` macro. We will utilize the `ExponentialFamilyProjection` library to project an arbitrary function onto a member of the exponential family. More information on `ExponentialFamilyProjection` can be found in the [Non-conjugate Inference](@ref inference-nonconjugate) section and in its [official documentation](https://github.com/ReactiveBayes/ExponentialFamilyProjection.jl).

```@example inference-undefinedrules
using ExponentialFamilyProjection

@constraints function projection_constraints()
    # Use `Beta` from `Distributions.jl` as it is compatible with the `ExponentialFamilyProjection` library
    q(p) :: ProjectedTo(Beta) 
end
```

With all components ready, we can proceed with the inference procedure:

```@example inference-undefinedrules
result = infer(
    model = simple_model(prior = BetaDistribution(1, 1), likelihood = BernoulliDistribution),
    data = (y = dataset, ),
    constraints = projection_constraints(),
    options = (
        rulefallback = NodeFunctionRuleFallback(),
    )
)
```

!!! note 
    For `rulefallback = NodeFunctionRuleFallback()` to function correctly, the node must be defined as `Stochastic` and the underlying object must be a subtype of `Distribution` from `Distributions.jl`.

### Result analysis

We can perform a simple analysis and compare the inferred value with the hidden value used to generate the actual dataset:

```@example inference-undefinedrules
using Plots, StatsPlots
using Test #hide
@test isapprox(hidden_p, mean(result.posteriors[:p]), atol=1e-2) #hide
plot(result.posteriors[:p], label = "posterior of p", fill = 0, fillalpha = 0.2)
vline!([ hidden_p ], label = "hidden p")
```

As shown, the estimated posterior is quite close to the actual hidden value of `p` used during the inference procedure.

## [Fusing deterministic transformations with stochastic nodes](@id inference-undefinedrules-fusedelta)

One of the limitations of the `NodeFunctionRuleFallback` implementation is that it does not support [`Deterministic` or `Delta` nodes](@ref delta-node-manual). However, it is possible to combine a deterministic transformation with a stochastic node, such as `Gaussian`. For instance, consider a dataset drawn from the `Normal` distribution, where the `mean` parameter has been transformed by a known function, and the true hidden variable is `h`.

```@example fusedelta
using ExponentialFamily, Distributions, Plots, StableRNGs

hidden_h = 2.3
hidden_t = 0.5

known_transformation(h) = exp(h)

hidden_mean = known_transformation(hidden_h)
ndatapoints = 50
dataset = rand(StableRNG(42), NormalMeanPrecision(hidden_mean, hidden_t), ndatapoints)

histogram(dataset; normalize = :pdf)
```

The model can be defined as follows:

```@example fusedelta
using RxInfer 

@model function mymodel(y, prior_h, prior_t)
    h ~ prior_h
    t ~ prior_t
    y .~ Normal(mean = known_transformation(h), precision = t)
end
```

Inference in this model is challenging because the `known_transformation` function is explicitly used as a factor node, requiring special approximation rules. These rules are covered in [a separate section](@ref delta-node-manual). Here, we demonstrate a different approach that modifies the model structure to run inference without needing to approximate messages around a deterministic node.

First, we define our custom transformed `Normal` distribution:

```@example fusedelta
using BayesBase

struct TransformedNormalDistribution{H, T} <: ContinuousUnivariateDistribution
    h::H
    t::T
end

# We integrate the `known_transformation` within the `logpdf` function
# This way, it won't be an explicit factor node but hidden within the `logpdf` of another node
BayesBase.logpdf(dist::TransformedNormalDistribution, x) = logpdf(NormalMeanPrecision(known_transformation(dist.h), dist.t), x)
BayesBase.insupport(dist::TransformedNormalDistribution, x) = true

@node TransformedNormalDistribution Stochastic [out, h, t]
```

Next, we tweak the model structure:

```@example fusedelta
@model function mymodel(y, prior_h, prior_t)
    h ~ prior_h
    t ~ prior_t
    y .~ TransformedNormalDistribution(h, t)
end
```

We use the following priors, [constraints](@ref user-guide-constraints-specification), and [initialization](@ref initialization):

```@example fusedelta
using ExponentialFamilyProjection

prior_h = LogNormal(0, 1)
prior_t = Gamma(1, 1)

constraints = @constraints begin
    q(h, t) = q(h)q(t)
    q(h) :: ProjectedTo(LogNormal)
    q(t) :: ProjectedTo(Gamma)
end

initialization = @initialization begin
    q(t) = Gamma(1, 1)
end
```

!!! note
    The `ProjectedTo` macro has a `parameters` field that allows for different hyperparameters, which may improve accuracy or convergence speed. Refer to the `ExponentialFamilyProjection` documentation for more information.

## Inference with a rule fallback

Now we are ready to run the inference procedure:

```@example fusedelta
result = infer(
    model = mymodel(prior_h = prior_h, prior_t = prior_t),
    data = (y = dataset,),
    constraints = constraints,
    initialization = initialization,
    iterations = 50,
    options = (
        rulefallback = NodeFunctionRuleFallback(),
    )
)
```

### Result analysis

Finally, let's plot the resulting posteriors for each VMP iteration:

```@example fusedelta 
using Test #hide
@test isapprox(mean(result.posteriors[:h][end]), hidden_h, atol=0.1) #hide

@gif for (i, q) in enumerate(zip(result.posteriors[:h], result.posteriors[:t]))
    p1 = plot(1:0.01:3, q[1], label = "q(h) iteration $i", fill = 0, fillalpha = 0.2)
    p1 = vline!([hidden_h], label = "hidden h")
    
    p2 = plot(0:0.01:1, q[2], label = "q(t) iteration $i", fill = 0, fillalpha = 0.2)
    p2 = vline!([hidden_t], label = "hidden t")
    
    plot(p1, p2)
end fps = 15
```

We can see that the inference results are able to recover the actual value of hidden `h` that has been used to generate the synthetic dataset. In conclusion, this example demonstrates that by integrating deterministic transformations within the logpdf function of a stochastic node, we can bypass the limitations of `NodeFunctionRuleFallback` in handling deterministic nodes.