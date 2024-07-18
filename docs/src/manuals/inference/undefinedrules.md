# [Inference without Explicit Message Update Rules](@id inference-undefinedrules)

`RxInfer` utilizes the [`ReactiveMP.jl`](https://github.com/ReactiveBayes/ReactiveMP.jl) package as its inference backend. Typically, running inference with `ReactiveMP.jl` requires users to define a factor node using the `@node` macro and specify corresponding message update rules with the `@rule` macro. Detailed instructions on this can be found in [this section](@ref create-node) of the documentation. However, in this tutorial, we will explore an alternative approach that allows inference with default message update rule for custom factor nodes by defining only `BayesBase.logpdf` and `BayesBase.insupport` for a factor node, without needing explicit `@rule` specifications.

!!! note 
    In the context of message-passing based Bayesian inference, custom message update rules enhance precision and efficiency. These rules leverage the specific mathematical properties of the model's distributions and relationships, leading to more accurate updates and faster convergence. By incorporating domain-specific knowledge, custom rules improve the robustness and reliability of the inference process, particularly in complex models where default rules may be inadequate or inefficient.

## A Simple Prior-Likelihood Model

Consider a simple model with a hidden variable `p` and observations `y`. We assume that `p` follows a prior distribution and `y` are drawn from a likelihood distribution. The model can be defined as follows:

```@example inference-undefinedrules
using RxInfer

@model function simple_model(y, prior, likelihood)
    p ~ prior
    y .~ likelihood(p)
end
```

## Node Specifications

Next, we define structures for both the `prior` and the `likelihood`. Let's start with the prior. Assume that the `p` parameter is best described by a `Beta` distribution. We can define it as follows:

!!! note
    The `Distributions.jl` package already defines `Beta` distributions. The example below redefines this structure for illustrative purposes.

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

!!! note
    Although `Beta` is a conjugate prior for the parameter of the `Bernoulli` distribution, `ReactiveMP` and `RxInfer` are unaware of this and cannot exploit this information. To utilize conjugacy, refer to the [custom node creation section](@ref create-node) of the documentation.

## Generating a Synthetic Dataset

We can generate a synthetic dataset as follows:

```@example inference-undefinedrules
using StableRNGs, Plots

hidden_p    = 1 / 3.14
ndatapoints = 1_000
dataset     = rand(StableRNG(42), Bernoulli(hidden_p), ndatapoints)

bar(["true", "false"], [ count(==(true), dataset), count(==(false), dataset) ], label = "dataset")
```

## Inference with Rule Fallback

Now, we can run inference with `RxInfer`. Since explicit rules for our nodes are not defined, we can instruct the `ReactiveMP` backend to use fallback message update rules. Refer to the `ReactiveMP` documentation for available fallbacks. In this example, we will use the `NodeFunctionRuleFallback` structure, which uses the `logpdf` of the stochastic node to approximate messages.

!!! note
    `NodeFunctionRuleFallback` employs a simple approximation for outbound messages, which may significantly degrade inference accuracy. Whenever possible, it is recommended to define [proper message update rules](@ref create-node).

To complete the inference setup, we must define an approximation method for posteriors using the `@constraints` macro. We will utilize the `ExponentialFamilyProjection` library to project an arbitrary function onto a member of the exponential family. More information on `ExponentialFamilyProjection` can be found in the [Non-conjugate Inference](@ref inference-nonconjugate) section.

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

### Result Analysis

We can perform a simple analysis and compare the inferred value with the hidden value used to generate the actual dataset:

```@example inference-undefinedrules
using Plots, StatsPlots
using Test #hide
@test isapprox(hidden_p, mean(result.posteriors[:p]), atol=1e-2) #hide
plot(result.posteriors[:p], label = "posterior of p", fill = 0, fillalpha = 0.2)
vline!([ hidden_p ], label = "hidden p")
```

As shown, the estimated posterior is quite close to the actual hidden value of `p` used during the inference procedure.