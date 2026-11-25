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
    For `rulefallback = NodeFunctionRuleFallback()` to apply to a `Stochastic` node, calling the node with its inputs, for example `BernoulliDistribution(p)`, must return an object that has a `logpdf`. A node with interface groups needs its own node function. For a `Deterministic` node, such as a Delta node without an approximation method, the fallback sends towards an input the message on the output pulled back through the node's function, and towards the output the function evaluated at the means of the inputs. The fallback does not apply to an input that is a joint marginal. If one of the node's rules fits, that rule is used instead of the fallback.

The option `default_form_constraint` sets a form constraint for every random variable that has no constraint of its own. Here `(q = …,)` applies it to the marginals only:

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

## [Universal inference](@id inference-undefinedrules-universal)

`NodeFunctionRuleFallback` replaces the other inputs of a node by their means, and it needs a projection for each variable. The `fallback` keyword of [`infer`](@ref) takes the distributions of the other inputs into account and sets the projections itself. It approximates the messages wherever the model has no exact rule, so no rules, approximation methods or initialization need to be written. Exact rules still run where they exist, and the projections leave closed forms unchanged, so a conjugate model under belief propagation gives the same result as without the keyword. Under a variational factorisation, the initialization from the priors and, for the schemes with sites, their schedule change the iterates but not the fixed point.

The approach relies on Gaussian beliefs. The beliefs of a variable are its marginal and the messages it sends, and each of them is a product of the messages the variable receives. When the beliefs about the neighbors of a node are Gaussian, the outgoing messages of any node with a log-density, and of any deterministic function, can be computed numerically. "Gaussian" is meant broadly here:

- A variable with a constrained domain, such as a precision, a rate or a probability, is Gaussian in unconstrained coordinates (log or logit), which are read from the support of its prior. Its posterior is reported on the original domain, for example as a log-normal distribution for a precision.
- A finite discrete variable stays categorical, and its expectations are exact sums.
- A deterministic node pulls the message on its output back through its function towards an input, and pushes its inputs forward through the function towards its output.

The approach does not cover discrete variables with unbounded support, such as a Poisson count that is not observed, or posteriors with several modes, which a single Gaussian cannot represent.

An approximation scheme makes three independent choices, and each `fallback` scheme fixes all three:

| Choice | Options |
|:-------|:--------|
| How messages are combined | each belief is projected separately, or each edge keeps one site fitted against its cavity (expectation propagation, natural-gradient propagation) |
| The form of a belief | a Gaussian from Laplace's method (mode and curvature), the variational Gaussian (minimizes `KL(q ‖ p)`) or moment matching; with natural-gradient sites, the variational optimum within the variable's own family, such as a Gamma or a Beta |
| How expectations are computed | closed forms, deterministic quadrature or sampling, through the expectation protocol of BayesBase |

### [Choosing a scheme](@id inference-undefinedrules-schemes)

| Scheme | Beliefs without a closed form | Free energy | Cost | Suited to |
|:-------|:------------------------------|:------------|:-----|:----------|
| [`LocalLaplace()`](@ref LocalLaplace) | Laplace approximation at the mode of each product of messages, each product separately | Bethe, with Laplace approximations of the normalizers | lowest | regression and hierarchical models; a first try |
| [`LaplacePropagation()`](@ref LaplacePropagation) | one site per edge, fitted by Laplace's method against the rest of the belief | expectation propagation, not a bound | a site per edge, a few iterations | state-space models and nonlinear transitions; on a chain it reaches the global Laplace approximation |
| [`ExpectationPropagation()`](@ref ExpectationPropagation) | one site per edge, fitted by moment matching against the rest of the belief | expectation propagation, not a bound | as Laplace propagation, times the number of quadrature points | state-space models where the posterior mean and variance matter more than the mode |
| [`NaturalGradientPropagation()`](@ref NaturalGradientPropagation) | one site per edge in the variable's own family (Gaussian, Gamma or Beta), updated by damped natural-gradient steps | Bethe, with average energies: the negative evidence lower bound on a Gaussian tree with non-conjugate leaves, otherwise a diagnostic | a site per edge, a few tens of iterations, a quadrature per step | skewed posteriors, positive and bounded variables, mean-field models, the variational Gaussian of a model |

Start with `LocalLaplace()`, the cheapest scheme, and switch to another one if it is not accurate enough.

On a skewed posterior, such as that of a variance estimated from little data, the Laplace approximation sits at the mode, while the variational and moment-matched Gaussians move towards the heavier tail. Under `NaturalGradientPropagation()`, a positive variable with a Gamma prior keeps a Gamma posterior.

Some messages have no Gaussian approximation on their own, for example a message pulled back through a nonlinear transition or observation function. They can only be approximated in the context of the rest of the belief, so use a cavity scheme for them. Projecting each belief separately with Laplace's method can lose track in such models, as a filter can at a turning point.

Only the free energy of `NaturalGradientPropagation()` is a bound on `-log p(y)`, and only when the model is a Gaussian tree whose non-conjugate factors are leaves, such as a regression with one latent vector, or under mean field. The free energies of the other schemes approximate `-log p(y)`, so model comparison based on them is approximate too, and a lower value does not always mean a better fit.

With `report = true`, `infer` records which posteriors were approximated, which nodes' messages formed them, whether each projection converged, and which parts of the result are likely to be inaccurate. See [What was approximated: the inference report](@ref user-guide-inference-report).

### [Laplace's approximation of each belief](@id inference-undefinedrules-laplace)

The projection reads the domain of a variable from the `minimum` and `maximum` of its prior, so we define them for `BetaDistribution`:

```@example inference-undefinedrules
Base.minimum(::BetaDistribution) = 0.0
Base.maximum(::BetaDistribution) = 1.0
nothing #hide
```

```@example inference-undefinedrules
result_laplace = infer(
    model = simple_model(prior = BetaDistribution(1, 1), likelihood = BernoulliDistribution),
    data = (y = dataset, ),
    fallback = LocalLaplace(),
)
@test isapprox(mean(result_laplace.posteriors[:p]), mean(Beta(1 + count(dataset), 1 + length(dataset) - count(dataset))), rtol=1e-3) #hide
mean(result_laplace.posteriors[:p]), std(result_laplace.posteriors[:p])
```

The posterior of `p` is a logit-normal distribution: a Gaussian in logit coordinates, mapped onto the unit interval. Its mean matches the mean of the exact posterior, `Beta(1 + 307, 1 + 693)`, to a relative tolerance of `1e-3`.

`LocalLaplace()` sets four options. Each of them can also be set directly, and an option set explicitly takes precedence over the scheme:

- The rule fallback (option `rulefallback`) is `DeltaMessagePassingRules.ExpectationRuleFallback()`. Its message is the expected log-density of the node, `x ↦ E_q[log f(x, ·)]`, with the expectation taken over all other inputs. It also computes the messages of Delta nodes without an approximation method, joint marginals and average energies, so the free energy can be computed.
- The default form constraint (option `default_form_constraint`) is `DeltaMessagePassingRules.LaplaceForm()`. It replaces every belief that has no closed form by its Laplace approximation, computed in the unconstrained coordinates of the variable's domain, and leaves closed forms unchanged. A message whose product has no mode, such as the message of a single likelihood, is left as it is (`LaplaceForm(nomode = :leave)`), so that exact rules downstream do not receive a Gaussian that Laplace's method could not compute.
- The expectation backend (option `context = (expectation = …,)`) defaults to the backends loaded in the session: the quadrature rules of ExpectationApproximations, which RxInfer loads, and the backends of any other package that registers them. Each call of `infer` collects them once into a type-stable router that tries exact backends first. `LocalLaplace(expectation = ExpectationApproximations.EADefault())` fixes the backend instead.
- The initialization strategy is [`FromPriors`](@ref), which initializes every variable that `initialization` leaves out from its prior. See [Initialization from priors](@ref initialization-from-priors).

The same inference with the options set explicitly:

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
LocalLaplace
RxInfer.FallbackScheme
RxInfer.fallback_rule
RxInfer.fallback_form
RxInfer.fallback_expectation
RxInfer.fallback_initialization
RxInfer.with_specification
```

### [Cavity propagation: Laplace propagation and expectation propagation](@id inference-undefinedrules-propagation)

`LocalLaplace()` projects each belief of a variable separately. The message to each neighbor and
the marginal are products of messages, and each product is projected on its own. A message with
no closed form, such as a likelihood message, is therefore approximated once for every product it
enters, each time in a different context. The resulting beliefs do not agree with one another,
and a message that has no Gaussian approximation by itself, such as one pulled back through a
nonlinear function, cannot be projected at all.

`fallback = LaplacePropagation()` and `fallback = ExpectationPropagation()` approximate each such
message once, in the context of the rest of the variable's belief. Every random variable keeps
one site per edge. The site of an edge is the edge's message multiplied by its cavity, projected,
and divided by the cavity again. The cavity of an edge is the product of the sites of the
variable's other edges, which is also the message the variable sends on that edge. Every belief
of the variable is then a product of the same sites, so the beliefs agree. A message that has a
closed form is its own site, so exact rules and conjugate products are unchanged.

`LaplacePropagation()` projects with Laplace's method, and `ExpectationPropagation()` with moment
matching. Moment matching uses Gauss–Hermite points placed on the Laplace approximation of the
tilted density, the message times the cavity. Points placed on the cavity would miss a message
that is much sharper than the cavity. `ExpectationPropagation(backend = SphericalRadialCubature())`,
with the backend from ExpectationApproximations, uses 2d points in d dimensions in place of a
product grid of `points`ᵈ points.

The message of a deterministic node towards its output is projected the same way, against the
cavity of the output. A node with several random inputs sends each of them the belief propagation
message: the node's factor integrated against the messages of the other inputs. The site of each
input is then its block of the joint projection.

The method is expectation propagation (Minka, 2001) with one Gaussian site per pair of variable
and node. For a node connected to a single variable, such as an observation, it is standard
expectation propagation. With Laplace's method in place of moment matching, it is Laplace
propagation (Smola, Vishwanathan and Eskin, 2003). The variables keep the sites, so the rules and
rule matching are unchanged, and a node with its own rule, such as the expectation propagation
rule of the `Probit` node, takes part as it is.

An initial message, `μ(x)` in [`@initialization`](@ref), is used as the first cavity of each edge
while the variable's other edges have no site yet. A variable whose edges all carry messages
without a closed form needs an initial message to start: each of these messages waits for a
cavity, and only a site on another edge can provide one. A loop through nodes with several random
inputs, as in loopy belief propagation, also needs one. The fixed point does not depend on the
initial message.

```julia
result = infer(
    model = state_space_model(),
    data = (y = observations,),
    iterations = 10,
    fallback = ExpectationPropagation(),
    free_energy = true,
)
```

A site is computed when the message on its edge arrives, so each iteration is one sweep over the
sites. On a chain, the sites settle within a few iterations. A message that arrives only once,
from a node whose inputs are all constants, has its site projected again at each iteration.

Sites, cavities and messages may be improper, that is, Gaussians with a singular or indefinite
precision. For example, an observation through `sin(θ)` carries no information about a velocity,
and gives a negative precision where `sin` curves against the cavity. Properness is not checked:
each site, cavity and message is passed on as computed, in natural parameters, which the rules of
the Gaussian nodes accept.

The free energy is that of expectation propagation. A node without rules contributes a term
computed from its inbound messages, the cavities, and all other terms are the Bethe terms. At a
fixed point it approximates `-log p(y)`, but it is not a bound: a lower value does not mean a
better fit, and the value can lie below `-log p(y)`. It is computed once per iteration, in batch
and in streaming inference, like the Bethe free energy, and it can be read in the same ways: the
stream, the histories, the `OnFreeEnergyUpdateEvent` callbacks and
[`StopEarlyIterationStrategy`](@ref).

The beliefs settle only to the precision of their projections, set by `tol`, so the free energy
keeps changing by about that much. Give [`StopEarlyIterationStrategy`](@ref) a tolerance well
above it, or it will not stop early.

```@docs
LaplacePropagation
ExpectationPropagation
RxInfer.fallback_site
RxInfer.SitePolicy
RxInfer.cavity_free_energy
```

### [Natural-gradient propagation](@id inference-undefinedrules-ngmp)

`fallback = NaturalGradientPropagation()` uses the same sites as the cavity schemes, one per edge
and fitted against the cavity, and the same belief propagation messages. It differs from them in
two ways (Lukashchuk et al., 2026).

First, a site belongs to the family of the variable's cavity: a Gaussian, a Gamma, a Beta or
another one-dimensional exponential family. A conjugate message of that family is its own site, so
a positive variable with a Gamma prior keeps a Gamma posterior.

Second, at each sweep a site moves by a damped natural-gradient step, where the cavity schemes
replace it by a projection. The target of the step is `η⋆ = G(q)⁻¹ Cov_q[T, ℓ]`, where `ℓ` is the
log of the edge's message, `q` is the marginal of the variable, `T` are the sufficient statistics
of its family and `G` is their Fisher information. In natural parameters, `η⋆` is the gradient of
`E_q[ℓ]` with respect to the mean parameters of `q`. The step is a heavy-ball update,
`v ← βv + α(η⋆ − η)`, `η ← η + v`, with `alpha = 0.5` and `beta = 0.2` by default, and it is
halved until the marginal is proper.

At a fixed point, each site equals its target, and each marginal is a stationary point of the
variational free energy over its family. With Gaussian sites, the method is
conjugate-computation variational inference (Khan and Lin, 2017). For a model with one latent
variable, it finds the variational Gaussian, the Gaussian that minimizes `KL(q ‖ p)` with respect
to the joint distribution. A belief without sites, of a variable with a single connection or whose
cavity is in no supported family, is projected onto its variational Gaussian with
`DeltaMessagePassingRules.VariationalForm`, in the unconstrained coordinates of its domain.

In the example below, the prior of `p` is `Beta` from Distributions.jl, so its message is a Beta
distribution. The `BernoulliDistribution` nodes have no rules and send raw messages, log-densities
without a closed form, and these get Beta sites. The Bernoulli log-likelihood is linear in the
sufficient statistics of the Beta family, so each site reaches it exactly and the posterior is the
conjugate one:

```@example inference-undefinedrules
result_ngmp = infer(
    model = simple_model(prior = Beta(1.0, 1.0), likelihood = BernoulliDistribution),
    data = (y = dataset, ),
    iterations = 40,
    fallback = NaturalGradientPropagation(),
)
posterior_ngmp = last(result_ngmp.posteriors[:p])
@test posterior_ngmp isa Beta #hide
@test isapprox(params(posterior_ngmp)[1], 1 + count(dataset), rtol = 1e-6) #hide
@test isapprox(params(posterior_ngmp)[2], 1 + length(dataset) - count(dataset), rtol = 1e-6) #hide
posterior_ngmp, Beta(1 + count(dataset), 1 + length(dataset) - count(dataset))
```

A site moves one step per sweep, so set `iterations` to a few tens. The report says whether the
last step on each edge had settled to `tol`, and warns with `:sites_not_converged` where it had
not. The first step for a raw message starts from a flat site and needs a proper cavity: a prior,
or an initial message `μ(x)` for a variable whose edges all carry raw messages. This includes the
predicted state of a filter with a nonlinear transition, `x_pred := f(x_prev)` followed by
`x ~ N(x_pred, Q)`. Nothing requests the message from `x ~ N(x_pred, Q)` back towards `x_pred`,
so the pushforward towards `x_pred` has no cavity until `μ(x_pred)` provides one. The family of a
variable is that of its prior or of its initial message, and does not depend on the order in which
its messages arrive. In streaming inference, the `iterations` of each data update continue from
the sites that the previous update left, so give each update enough sweeps for its sites to
settle.

Sites in families other than the Gaussian are fitted to the same messages that the cavity schemes
send: the log-density of a node, the pushforward of a deterministic node, such as the log-normal
message of `exp` towards a Gamma variable, the belief propagation message of a node with several
Gaussian inputs, or a closed-form message of another family. A target in a Gaussian family is
computed with 16 Gauss–Hermite points per dimension, and a target in another family with 128
Gauss–Legendre points in the unconstrained coordinate of its domain.

With [ClosedFormExpectations](https://github.com/biaslab/ClosedFormExpectations.jl) loaded
(`using ClosedFormExpectations`), a target is computed in closed form, without quadrature,
wherever that package has a closed form for the message and the variable's family. This covers
several of the messages above whose log-density is that of a distribution: the message of a
Poisson count pulled back through `exp` towards its log-rate, with a Gaussian site; a log-normal
message, with a Gamma site; and a Laplace likelihood, with a Gaussian site, whose kink makes
quadrature converge slowly. Other targets still use quadrature. On smooth messages, the closed form and
the quadrature agree to about `1e-9`, and the closed form is faster.
NaturalGradientMessagePassingRules depends on ClosedFormExpectations, so its rules always use
these closed forms.

The free energy is the Bethe free energy, in which each node without rules contributes its average
energy `-E_q[log f]` under the marginals. Its stationary point is the fixed point of the
iterations. On a Gaussian tree whose only non-conjugate factors are leaves, such as a regression,
it is the negative evidence lower bound. Elsewhere it is a diagnostic, and it need not decrease
from one iteration to the next.

The momentum is carried from one marginal to the next without the vector transport of information
geometry, and the iteration has no convergence guarantee. The default `alpha` and `beta` damp the
steps, and halving a step keeps every marginal proper on long chains. When the other inputs of a
node are not all Gaussian messages or point masses, for example when one of them is a Gamma
precision, the message towards an input is the expected log-density, and a step on it is a step of
non-conjugate variational message passing.

```@docs
NaturalGradientPropagation
RxInfer.NaturalGradientSitePolicy
```

### [Limits](@id inference-undefinedrules-limits)

- Under the cavity schemes, a node with several random inputs sends the belief propagation
  message only when its other inputs are Gaussian messages or point masses. When another input
  carries a different message, such as a Gamma precision, or arrives as a mean-field marginal, the
  node sends the expected log-density instead, which approximates the belief propagation message.
  The report flags this as `:expectation_over_messages`.
- The sites of Laplace propagation and expectation propagation are Gaussian. The beliefs of a
  variable in another family, such as a Gamma, are projected product by product, as under
  `LocalLaplace()`. `NaturalGradientPropagation()` gives such a variable sites in its own family.
- The sites of Laplace propagation and expectation propagation are not damped. On the models
  tried, updating the sites as their messages arrive converged without damping. Natural-gradient
  sites are damped.
- Laplace's method is centered at the mode. On a skewed posterior, its Gaussian sits at the mode
  under both `LocalLaplace()` and `LaplacePropagation()`, while the variational and moment-matched
  Gaussians move away from it.

### [References](@id inference-undefinedrules-references)

- T. P. Minka (2001). Expectation propagation for approximate Bayesian inference. *UAI*.
- T. P. Minka (2005). Divergence measures and message passing. Microsoft Research, MSR-TR-2005-173.
- A. J. Smola, S. V. N. Vishwanathan and E. Eskin (2003). Laplace propagation. *NeurIPS*.
- M. E. Khan and W. Lin (2017). Conjugate-computation variational inference. *AISTATS*.
- M. Lukashchuk, K. Yemets, A. Ledbetter and İ. Şenöz (2026). Information geometry of message passing. arXiv:2608.15922.
- C. Wang and D. M. Blei (2013). Variational inference in nonconjugate models. *JMLR* 14.
- A. Vehtari et al. (2020). Expectation propagation as a way of life. *JMLR* 21(17).

## [Fusing deterministic transformations with stochastic nodes](@id inference-undefinedrules-fusedelta)

For a [`Deterministic` or `Delta` node](@ref delta-node-manual), `NodeFunctionRuleFallback` pulls the message on the output back towards an input, but towards the output it sends the function evaluated at the means of the inputs, which discards their uncertainty. Besides the universal schemes above, another option is to combine the deterministic transformation with the stochastic node that follows it, such as a `Gaussian`. For instance, consider a dataset drawn from the `Normal` distribution, where the `mean` parameter has been transformed by a known function, and the true hidden variable is `h`.

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

Here `known_transformation` becomes a separate Delta node, whose messages need either an approximation method, covered in [a separate section](@ref delta-node-manual), or one of the universal schemes above. Instead, we change the model so that no message passes through a deterministic node.

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