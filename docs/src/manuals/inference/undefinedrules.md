# [Inference without explicit message update rules](@id inference-undefinedrules)

`RxInfer` utilizes the [`ReactiveMP.jl`](https://github.com/ReactiveBayes/ReactiveMP.jl) package as its inference backend. Typically, running inference with a custom factor node requires you to declare the node with the [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node) macro and to define its message update rules with the [`@define_message_update_rule`](@extref MessagePassingRulesBase.@define_message_update_rule) macro. For background on what rules are and how they work, see [Understanding Rules](@ref what-is-a-rule). Detailed instructions on implementing rules can be found in [this section](@ref create-node) of the documentation. In this tutorial, we explore an alternative approach: inference with a fallback message update rule for custom factor nodes, which only needs `BayesBase.logpdf` and `BayesBase.insupport` for the node's distribution, without any rules.

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

The next step is to declare these structures as factor nodes:

```@example inference-undefinedrules
@define_factor_node(node = BetaDistribution, type = Stochastic, interfaces = [:out, :a, :b])
@define_factor_node(node = BernoulliDistribution, type = Stochastic, interfaces = [:out, :p])
```

When declaring a node for a custom distribution, you must follow a specific interface ordering. The first interface is always `out`, which represents a sample in the `logpdf` function. All remaining interfaces must match the parameters of the distribution's constructor in the exact same order. For example, for the `BetaDistribution`, the node function is `(out, a, b) -> logpdf(BetaDistribution(a, b), out)`. The declaration defines this function, [`nodefunction`](@extref MessagePassingRulesBase.nodefunction), which the fallback evaluates:

```@example inference-undefinedrules
f = MessagePassingRulesBase.nodefunction(BernoulliDistribution)
f(out = true, p = 0.3) ≈ logpdf(Bernoulli(0.3), true)
```

!!! note
    Although `Beta` is a conjugate prior for the parameter of the `Bernoulli` distribution, `ReactiveMP` and `RxInfer` are unaware of this and cannot exploit this information. To utilize conjugacy, refer to the [custom node creation section](@ref create-node) of the documentation.

## Generating a synthetic dataset

Previously, we assumed that our dataset consists of discrete values: `true` and `false`. We can generate a synthetic dataset with these values as follows:

```@example inference-undefinedrules
using StableRNGs, Plots

hidden_p    = 1 / 3.1415 # a value between `0` and `1`
ndatapoints = 1_000      # number of observations
dataset     = rand(StableRNG(42), Bernoulli(hidden_p), ndatapoints)

bar(["true", "false"], [ count(==(true), dataset), count(==(false), dataset) ], label = "dataset")
```

## Inference with a rule fallback

Now, we can run inference with `RxInfer`. Since explicit rules for our nodes have not been defined, we can instruct the `ReactiveMP` backend to use a fallback where no rule fits, with the `rulefallback` option. Refer to the [rule fallbacks](@extref MessagePassingRulesBase Rule-fallbacks) page of MessagePassingRulesBase for how fallbacks work. In this example, we use [`NodeFunctionRuleFallback`](@extref MessagePassingRulesBase.NodeFunctionRuleFallback), which uses the `logpdf` of the stochastic node to approximate messages: its message towards an interface is the node's log-density as a function of that interface, with every other input collapsed to its mean.

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
    For `rulefallback = NodeFunctionRuleFallback()` to apply, calling a `Stochastic` node with its inputs, as `BernoulliDistribution(p)`, must return an object with a `logpdf`; a node with interface groups needs a node function of its own. A `Deterministic` node, such as a Delta node without an approximation method, sends the message on its output pulled back through its function towards an input, and its function of the inputs' means towards its output. The fallback does not apply to an input that is a joint marginal, and a rule of the node, where one fits, takes precedence over it.

The constraint can also be given once for every random variable that has none of its own, with the option `default_form_constraint`; `(q = …,)` applies it to the marginals only:

```@example inference-undefinedrules
result_by_default = infer(
    model = simple_model(prior = BetaDistribution(1, 1), likelihood = BernoulliDistribution),
    data = (y = dataset, ),
    options = (
        rulefallback = NodeFunctionRuleFallback(),
        default_form_constraint = (q = ProjectedTo(Beta),),
    )
)
```

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

## [Universal Laplace inference](@id inference-undefinedrules-laplace)

`NodeFunctionRuleFallback` collapses every other input to its mean, and needs a projection chosen per variable. One keyword of [`infer`](@ref), `fallback = LaplaceApproximation()`, needs neither: wherever the model has no exact rule, it approximates by Laplace's method, with no method per node or variable.

The projection reads a variable's domain from its prior's `minimum` and `maximum`, so the prior says them:

```@example inference-undefinedrules
Base.minimum(::BetaDistribution) = 0.0
Base.maximum(::BetaDistribution) = 1.0
nothing #hide
```

```@example inference-undefinedrules
result_laplace = infer(
    model = simple_model(prior = BetaDistribution(1, 1), likelihood = BernoulliDistribution),
    data = (y = dataset, ),
    fallback = LaplaceApproximation(),
)
@test isapprox(mean(result_laplace.posteriors[:p]), mean(Beta(1 + count(dataset), 1 + length(dataset) - count(dataset))), rtol=1e-3) #hide
mean(result_laplace.posteriors[:p]), std(result_laplace.posteriors[:p])
```

The posterior of `p` is a logit-normal distribution, a Gaussian in logit coordinates pushed forward onto the unit interval. Its mean is that of the exact posterior, `Beta(1 + 307, 1 + 693)` by conjugacy, to a relative `1e-3`. A rule that exists still takes precedence over the fallback, and a closed form passes the projection, so a conjugate model is inferred as without the keyword.

`LaplaceApproximation()` sets four things, each of which the options can set instead:

- the rule fallback, `DeltaMessagePassingRules.ExpectationRuleFallback()`, the option `rulefallback`. It sends a node's expected log-density, `x ↦ E_q[log f(x, ·)]` over every other input. It also serves Delta nodes without an approximation method, joint marginals and average energies, so the free energy is available;
- the default form constraint, `DeltaMessagePassingRules.LaplaceForm()`, the option `default_form_constraint`. It projects every belief with no closed form by Laplace's approximation, in the unconstrained coordinates of the variable's domain, and leaves closed forms as they are. A message whose product has no mode, such as a likelihood's message alone, is left as it is, `LaplaceForm(nomode = :leave)`, so that an exact rule downstream is never fed a Gaussian Laplace did not find;
- the expectation backend, `context = (expectation = …,)`. By default, the backends loaded in the session, frozen once per call of `infer` into a type-stable router that takes exact ones first: ExpectationApproximations' quadrature rules, which RxInfer loads, and any other package's that registers its own. `LaplaceApproximation(expectation = ExpectationApproximations.EADefault())` pins one;
- the initialization, [`FromPriors`](@ref), which starts every variable the `initialization` leaves out from its prior. See [Initialization from priors](@ref initialization-from-priors).

`report = true` records what the keyword approximated and how: which posteriors, by which nodes' messages, whether each Laplace approximation converged, and what is likely to be inaccurate. See [What was approximated: the inference report](@ref user-guide-inference-report).

The same inference with the options:

```julia
using ExpectationApproximations

infer(
    model = simple_model(prior = BetaDistribution(1, 1), likelihood = BernoulliDistribution),
    data = (y = dataset, ),
    initialization = FromPriors(),
    options = (
        rulefallback = DeltaMessagePassingRules.ExpectationRuleFallback(),
        default_form_constraint = (q = DeltaMessagePassingRules.LaplaceForm(), μ = DeltaMessagePassingRules.LaplaceForm(nomode = :leave)),
        context = (expectation = ExpectationApproximations.EADefault(),),
    ),
)
```

```@docs
LaplaceApproximation
RxInfer.FallbackScheme
RxInfer.fallback_rule
RxInfer.fallback_form
RxInfer.fallback_expectation
RxInfer.fallback_initialization
RxInfer.with_specification
```

### [Conjugate-computation variational inference](@id inference-undefinedrules-cvi)

`fallback = CVI()` keeps the same rules and changes the projection. A belief with no closed form
becomes the Gaussian that minimises `KL(q ‖ p)`, in the unconstrained coordinates of its domain,
instead of Laplace's Gaussian at the mode: `DeltaMessagePassingRules.VariationalForm` in place of
`LaplaceForm`.
- Under mean field, a variable's belief is the product of its prior and the rule fallback's
  expected log-densities, and this Gaussian is the optimal one. It is the fixed point of the
  natural gradient of conjugate-computation variational inference.
- The free energy is then the negative evidence lower bound, never above Laplace's.
- The expectations of the projection go to the same backend as those of the rules.

```@example inference-undefinedrules
result_cvi = infer(
    model = simple_model(prior = BetaDistribution(1, 1), likelihood = BernoulliDistribution),
    data = (y = dataset, ),
    fallback = CVI(),
)
@test isapprox(mean(result_cvi.posteriors[:p]), mean(Beta(1 + count(dataset), 1 + length(dataset) - count(dataset))), rtol=1e-3) #hide
mean(result_cvi.posteriors[:p]), std(result_cvi.posteriors[:p])
```

With a thousand observations, the posterior is close to Gaussian in logit coordinates, so both
schemes agree. On a skewed posterior the variational Gaussian is shifted towards the heavier tail,
where Laplace's stays at the mode.

```@docs
CVI
```

### [Cavity propagation: Laplace propagation and expectation propagation](@id inference-undefinedrules-propagation)

`LaplaceApproximation()` and `CVI()` project each belief of a variable separately: the message to
each neighbour and the marginal are each a product of messages, projected on its own. A message
with no closed form, such as a likelihood's, is then approximated once per product, each time in
a different context. The beliefs disagree with one another, and a message that has no Gaussian of
its own, such as one pulled back through a nonlinear function, cannot be projected at all.

`fallback = LaplacePropagation()` and `fallback = ExpectationPropagation()` approximate each such
message once, against its context.
- Every random variable keeps one site per edge: the edge's message times its cavity, projected,
  divided by the cavity.
- The cavity is the product of the other edges' sites, which is the variable's message out on that
  edge.
- Every belief of the variable is then a product of sites, one Gaussian seen from each edge.
- The projection is Laplace's method for `LaplacePropagation()` and moment matching for
  `ExpectationPropagation()`, with Gauss–Hermite points on the tilted density's own Laplace
  approximation.
- A deterministic node's message towards its output is projected the same way, against the
  output's cavity.

```julia
result = infer(
    model = state_space_model(),
    data = (y = observations,),
    iterations = 10,
    fallback = ExpectationPropagation(),
    free_energy = true,
)
```

**Iterations and proper-ness.**
- A site is computed when its edge's message arrives, so each iteration is a sweep over the sites;
  a chain settles in a few.
- Sites, cavities and messages may be improper, a Gaussian with a singular or an indefinite
  precision. Nothing checks: each is passed as it is computed, in natural parameters, which the
  Gaussian nodes' rules take.

**The free energy** is expectation propagation's.
- A node without rules takes its term from its inbound messages, the cavities, and every other
  term is Bethe's.
- It is computed once an iteration, in batch and in streaming inference, as Bethe's is, and every
  way of reading it works as without sites: the stream, the histories, the
  `OnFreeEnergyUpdateEvent` callbacks and [`StopEarlyIterationStrategy`](@ref).

```@docs
LaplacePropagation
ExpectationPropagation
RxInfer.fallback_site
RxInfer.SitePolicy
```

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

@define_factor_node(node = TransformedNormalDistribution, type = Stochastic, interfaces = [:out, :h, :t])
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