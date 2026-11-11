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

## [Universal inference](@id inference-undefinedrules-universal)

`NodeFunctionRuleFallback` collapses every other input to its mean, and needs a projection chosen per variable. One keyword of [`infer`](@ref), `fallback`, needs neither: wherever the model has no exact rule, it approximates, with no rule, method or initialization to write. Where an exact rule exists, it runs unchanged, so a conjugate model is inferred as without the keyword.

The idea is that message passing can be automated if every belief is Gaussian. A variable's beliefs are its marginal and the messages it sends, each a product of the messages it receives. A node of any kind is then asked one question: given Gaussian beliefs about its neighbors, what should it send? Every node with a log-density, and every deterministic function, can answer it numerically. "Gaussian" is meant broadly:

- a variable with a constrained domain, such as a precision, a rate or a probability, is Gaussian in unconstrained coordinates (log, logit), read from its prior's support. The posterior is reported on the original domain, a log-normal for a precision;
- a finite discrete variable stays categorical, its expectations exact sums;
- a deterministic node pulls the message on its output back through its function, towards an input, and pushes its inputs forward, towards its output.

Out of scope: discrete variables with unbounded support, such as a Poisson count that is not observed, and posteriors with several modes, which one Gaussian cannot represent.

An approximation scheme is three independent choices, and each `fallback` scheme makes them:

| Choice | Options |
|:-------|:--------|
| How messages combine | each belief projected on its own, or one site per edge against its cavity (expectation propagation, natural-gradient propagation) |
| The form of a belief | a Gaussian: Laplace's (mode and curvature), the variational one (minimizes `KL(q ‖ p)`), or moment matching; or, with natural-gradient sites, the variational optimum in the variable's own family, such as a Gamma or a Beta |
| How expectations are computed | exact closed forms, deterministic quadrature, or sampling, through BayesBase's expectation protocol |

### [Choosing a scheme](@id inference-undefinedrules-schemes)

| Scheme | Each belief without a closed form | Free energy | Cost | Use it for |
|:-------|:----------------------------------|:------------|:-----|:-----------|
| [`LocalLaplace()`](@ref LocalLaplace) | Laplace's Gaussian at the mode of the product, each product on its own | Bethe's, with Laplace's normalizers | the lowest | regression and hierarchical models, a first try |
| [`LaplacePropagation()`](@ref LaplacePropagation) | one site per edge by Laplace's method, against the rest of the belief | expectation propagation's, not a bound | a site per edge, a few iterations | state-space models and nonlinear transitions; reaches the global Laplace approximation on a chain |
| [`ExpectationPropagation()`](@ref ExpectationPropagation) | one site per edge by moment matching, against the rest of the belief | expectation propagation's, not a bound | as Laplace propagation, times the quadrature points | state-space models where the posterior's mean and variance matter more than its mode |
| [`NaturalGradientPropagation()`](@ref NaturalGradientPropagation) | one site per edge in the variable's own family, a Gaussian, a Gamma or a Beta, moved by damped natural-gradient steps | Bethe's with average energies: the negative evidence lower bound on a Gaussian tree with non-conjugate leaves, a diagnostic elsewhere | a site per edge, a few tens of iterations, a quadrature per step | skewed posteriors, positive and bounded variables, mean-field models, the variational Gaussian of a model |

- **Start with `LocalLaplace()`.** Where it is accurate enough, it is the cheapest.
- **A skewed posterior**, such as a variance with little data: Laplace's Gaussian sits at the mode, and the variational and moment-matched Gaussians move towards the heavier tail. Under `NaturalGradientPropagation()` a positive variable with a Gamma prior keeps a Gamma posterior.
- **A message without a Gaussian of its own**, such as one pulled back through a nonlinear transition or observation function, needs the context of the rest of the belief: use a cavity scheme. Laplace's approximation of each belief on its own can lose track there, as a filter does at a turning point.
- **The free energy.** Only `NaturalGradientPropagation()`'s, where the model is a Gaussian tree whose non-conjugate factors are leaves, such as a regression with one latent vector, or under mean field, is a bound on `-log p(y)`. The others approximate it, so comparing models by them is approximate, and a lower value is not always a better fit.

`report = true` records what the keyword approximated and how: which posteriors, by which nodes' messages, whether each projection converged, and what is likely to be inaccurate. See [What was approximated: the inference report](@ref user-guide-inference-report).

### [Laplace's approximation of each belief](@id inference-undefinedrules-laplace)

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
    fallback = LocalLaplace(),
)
@test isapprox(mean(result_laplace.posteriors[:p]), mean(Beta(1 + count(dataset), 1 + length(dataset) - count(dataset))), rtol=1e-3) #hide
mean(result_laplace.posteriors[:p]), std(result_laplace.posteriors[:p])
```

The posterior of `p` is a logit-normal distribution, a Gaussian in logit coordinates pushed forward onto the unit interval. Its mean is that of the exact posterior, `Beta(1 + 307, 1 + 693)` by conjugacy, to a relative `1e-3`. A rule that exists still takes precedence over the fallback, and a closed form passes the projection, so a conjugate model is inferred as without the keyword.

`LocalLaplace()` sets four things, each of which the options can set instead:

- the rule fallback, `DeltaMessagePassingRules.ExpectationRuleFallback()`, the option `rulefallback`. It sends a node's expected log-density, `x ↦ E_q[log f(x, ·)]` over every other input. It also serves Delta nodes without an approximation method, joint marginals and average energies, so the free energy is available;
- the default form constraint, `DeltaMessagePassingRules.LaplaceForm()`, the option `default_form_constraint`. It projects every belief with no closed form by Laplace's approximation, in the unconstrained coordinates of the variable's domain, and leaves closed forms as they are. A message whose product has no mode, such as a likelihood's message alone, is left as it is, `LaplaceForm(nomode = :leave)`, so that an exact rule downstream is never fed a Gaussian Laplace did not find;
- the expectation backend, `context = (expectation = …,)`. By default, the backends loaded in the session, frozen once per call of `infer` into a type-stable router that takes exact ones first: ExpectationApproximations' quadrature rules, which RxInfer loads, and any other package's that registers its own. `LocalLaplace(expectation = ExpectationApproximations.EADefault())` pins one;
- the initialization, [`FromPriors`](@ref), which starts every variable the `initialization` leaves out from its prior. See [Initialization from priors](@ref initialization-from-priors).

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
LocalLaplace
RxInfer.FallbackScheme
RxInfer.fallback_rule
RxInfer.fallback_form
RxInfer.fallback_expectation
RxInfer.fallback_initialization
RxInfer.with_specification
```

### [Cavity propagation: Laplace propagation and expectation propagation](@id inference-undefinedrules-propagation)

`LocalLaplace()` projects each belief of a variable separately: the message to
each neighbor and the marginal are each a product of messages, projected on its own. A message
with no closed form, such as a likelihood's, is then approximated once per product, each time in
a different context. The beliefs disagree with one another, and a message that has no Gaussian of
its own, such as one pulled back through a nonlinear function, cannot be projected at all.

`fallback = LaplacePropagation()` and `fallback = ExpectationPropagation()` approximate each such
message once, against its context.
- Every random variable keeps one site per edge: the edge's message times its cavity, projected,
  divided by the cavity.
- The cavity is the product of the other edges' sites, which is the variable's message out on that
  edge.
- Every belief of the variable is then a product of sites, one Gaussian seen from each edge, so
  the beliefs agree. A message that has a closed form is its own site, so exact rules and
  conjugate products are unchanged.
- The projection is Laplace's method for `LaplacePropagation()` and moment matching for
  `ExpectationPropagation()`, with Gauss–Hermite points on the tilted density's own Laplace
  approximation. Points placed on the cavity instead would miss a message much sharper than it.
  `ExpectationPropagation(backend = SphericalRadialCubature())`, from ExpectationApproximations,
  takes 2d points in d dimensions instead of a product grid of `points`ᵈ.
- A deterministic node's message towards its output is projected the same way, against the
  output's cavity.
- A node with several random inputs sends each of them belief propagation's message, the node's
  factor integrated against the other inputs' messages, and the input's site is its block of
  their joint projection.

This is expectation propagation (Minka, 2001) with one Gaussian site per variable and node: for a
node of one variable, such as an observation, it is expectation propagation exactly. Under
Laplace's method in place of moment matching, it is Laplace propagation (Smola, Vishwanathan and
Eskin, 2003). The sites live at the variables, not in the nodes, so rules and rule matching are
unchanged, and a node with a rule of its own, such as Probit's expectation propagation rule,
takes part as it is.

**Initial messages.** An initial message, `μ(x)` in [`@initialization`](@ref), is also each
edge's first cavity, while the variable's other edges have no site. A variable whose edges all
carry messages without a closed form needs one to start: each such message waits for a cavity,
which only another edge's site gives. So does a loop through nodes with several random inputs, as
in loopy belief propagation. The fixed point does not depend on it.

```julia
result = infer(
    model = state_space_model(),
    data = (y = observations,),
    iterations = 10,
    fallback = ExpectationPropagation(),
    free_energy = true,
)
```

**Iterations and properness.**
- A site is computed when its edge's message arrives, so each iteration is a sweep over the sites;
  a chain settles in a few. A message that arrives only once, from a node whose inputs are all
  constants, has its site projected again each iteration.
- Sites, cavities and messages may be improper, a Gaussian with a singular or an indefinite
  precision: an observation through `sin(θ)` says nothing about a velocity, and gives a negative
  precision where `sin` curves against the cavity. Nothing checks: each is passed as it is computed, in
  natural parameters, which the Gaussian nodes' rules take.

**The free energy** is expectation propagation's.
- A node without rules takes its term from its inbound messages, the cavities, and every other
  term is Bethe's.
- It approximates `-log p(y)` at a fixed point, but it is not a bound: a lower value is not a
  better fit, and it can lie below `-log p(y)`.
- It is computed once an iteration, in batch and in streaming inference, as Bethe's is, and every
  way of reading it works as without sites: the stream, the histories, the
  `OnFreeEnergyUpdateEvent` callbacks and [`StopEarlyIterationStrategy`](@ref).
- The beliefs settle only to the precision of their projections, set by their `tol`, so the free
  energy keeps moving by about that much. Give [`StopEarlyIterationStrategy`](@ref) a tolerance
  well above it, or it never stops early.

```@docs
LaplacePropagation
ExpectationPropagation
RxInfer.fallback_site
RxInfer.SitePolicy
RxInfer.cavity_free_energy
```

### [Natural-gradient propagation](@id inference-undefinedrules-ngmp)

`fallback = NaturalGradientPropagation()` keeps the sites, one per edge against its cavity, and the
messages of the cavity schemes, belief propagation's. It changes two things (Lukashchuk et al.,
2026):
- **A site lives in the variable's own family**, the family of its cavity: a Gaussian, a Gamma,
  a Beta, any exponential family of one dimension. A conjugate message of that family is its own
  site, so a positive variable with a Gamma prior keeps a Gamma posterior.
- **A site moves by a damped natural-gradient step** at each sweep, rather than to a projection.
  Its target is `η⋆ = G(q)⁻¹ Cov_q[T, ℓ]`, with `ℓ` the edge's log-message, `q` the variable's
  marginal, `T` its family's sufficient statistics and `G` their Fisher information, in natural
  parameters: the gradient of `E_q[ℓ]` in the mean parameters of `q`. The step is the heavy ball
  `v ← βv + α(η⋆ − η)`, `η ← η + v`, with `alpha = 0.5` and `beta = 0.2` by default, halved until
  the marginal is proper.

At a fixed point each site is its target, and each marginal a stationary point of the variational
free energy over its family. Under Gaussian sites this is conjugate-computation variational
inference (Khan and Lin, 2017): the variational Gaussian, the one that minimizes `KL(q ‖ p)` of
the joint, for a model with one latent variable. A belief with no sites, of a variable with one
connection or whose cavity is in no family, is projected onto its variational Gaussian,
`DeltaMessagePassingRules.VariationalForm`, in the unconstrained coordinates of its domain.

Here the Beta prior of `p` is a distribution, so its message is a Beta, and every Bernoulli
message, raw, takes a Beta site. The Bernoulli log-likelihood is of the Beta family, so each site
reaches it exactly, and the posterior is the conjugate one:

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

**Iterations.** A site moves one step per sweep, so give `iterations`, a few tens. The report
says whether each edge's last step had settled to `tol`, and warns with `:sites_not_converged`
where it had not. A raw message's first step starts from the flat site, against its cavity, which
must be proper: a prior, or an initial message `μ(x)` for a variable whose edges are all raw. That
includes a filter's predicted state through a nonlinear transition, `x_pred := f(x_prev)` before
`x ~ N(x_pred, Q)`: nothing asks for the message back towards it, so its pushforward has no cavity
until `μ(x_pred)` gives it one. A variable's family is that of its prior or its initial message,
never of the order in which its messages arrive. In streaming inference, each data update's `iterations` move the sites on from
where the last update left them: give each update enough sweeps for its sites to settle.

**Non-Gaussian sites** come from the messages the cavity schemes send: a node's log-density, a
deterministic node's pushforward, the log-normal of `exp` towards a Gamma, belief propagation's
message of a node with several Gaussian inputs, or a closed message of another family. A target in
a Gaussian family uses Gauss–Hermite points, and in another family Gauss–Legendre points in the
unconstrained coordinate of its domain.

**The free energy** is Bethe's, with the average energy `-E_q[log f]` of each node without rules
under the marginals, not expectation propagation's: its stationary point is the fixed point. On a
Gaussian tree whose only non-conjugate factors are leaves, such as a regression, it is the
negative evidence lower bound; elsewhere it is a diagnostic, which need not decrease from one
iteration to the next.

**Limits.** The momentum is carried from one marginal to the next without the vector transport of
information geometry, and the iteration has no convergence guarantee: the defaults are damped,
and the halving keeps every marginal proper on long chains. Towards an input of a node whose other
inputs are not all Gaussian messages or point masses, such as a Gamma precision, the message is
the expected log-density, and one step on it is non-conjugate variational message passing.

```@docs
NaturalGradientPropagation
RxInfer.NaturalGradientSitePolicy
```

### [Limits](@id inference-undefinedrules-limits)

- **A node with several random inputs** sends belief propagation's message under the cavity
  schemes only when its other inputs are Gaussian messages or point masses. An input with another
  message, such as a Gamma precision, or one that comes as a mean-field marginal, still gets the
  expected log-density, an approximation of that message. The report flags it as
  `:expectation_over_messages`.
- **Laplace and expectation propagation's sites are Gaussian.** A variable whose belief is
  another family, such as a Gamma, has its beliefs projected per product, as under
  `LocalLaplace()`; `NaturalGradientPropagation()` gives it sites of its own family.
- **Laplace and expectation propagation's sites are not damped.** On the models tried, sweeping
  the sites as their messages arrive converged without it. Natural-gradient sites are damped.
- **Laplace's method is mode-seeking.** On a skewed posterior it sits at the mode, under
  `LocalLaplace()` and `LaplacePropagation()` alike; the variational and moment-matched Gaussians
  do not.

### [References](@id inference-undefinedrules-references)

- T. P. Minka (2001). Expectation propagation for approximate Bayesian inference. *UAI*.
- T. P. Minka (2005). Divergence measures and message passing. Microsoft Research, MSR-TR-2005-173.
- A. J. Smola, S. V. N. Vishwanathan and E. Eskin (2003). Laplace propagation. *NeurIPS*.
- M. E. Khan and W. Lin (2017). Conjugate-computation variational inference. *AISTATS*.
- M. Lukashchuk, K. Yemets, A. Ledbetter and İ. Şenöz (2026). Information geometry of message passing. arXiv:2608.15922.
- C. Wang and D. M. Blei (2013). Variational inference in nonconjugate models. *JMLR* 14.
- A. Vehtari et al. (2020). Expectation propagation as a way of life. *JMLR* 21(17).

## [Fusing deterministic transformations with stochastic nodes](@id inference-undefinedrules-fusedelta)

Through a [`Deterministic` or `Delta` node](@ref delta-node-manual), `NodeFunctionRuleFallback` pulls the message on the output back towards an input, but sends the function of the inputs' means towards the output, which loses their uncertainty. Another way, without the universal schemes above, is to combine the deterministic transformation with the stochastic node that follows it, such as a `Gaussian`. For instance, consider a dataset drawn from the `Normal` distribution, where the `mean` parameter has been transformed by a known function, and the true hidden variable is `h`.

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

Here `known_transformation` is a factor node of its own, a Delta node, whose messages need an approximation method, covered in [a separate section](@ref delta-node-manual), or one of the universal schemes above. Instead, we change the model so that no message passes through a deterministic node.

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