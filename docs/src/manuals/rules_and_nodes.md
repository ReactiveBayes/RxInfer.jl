# [Understanding rules](@id what-is-a-rule)

A [rule](@extref MessagePassingRulesBase glossary-rule) computes one message of one
[factor node](@extref MessagePassingRulesBase glossary-factor-node): the message towards one of
the node's [interfaces](@extref MessagePassingRulesBase glossary-interface), from what the node
receives on the others. RxInfer finds a rule for every message a model needs. This page explains
how it finds one, and why some rules take
[messages](@extref MessagePassingRulesBase glossary-message) while others take
[marginals](@extref MessagePassingRulesBase glossary-marginal).

[Messages by hand](@ref learning-messages-by-hand) and
[Variational message passing by hand](@ref learning-vmp-by-hand) call the rules of small models
one by one, and are the practical companion to this page.

```@example rules
using RxInfer
nothing # hide
```

## What a rule computes

Consider a node ``f(x, y, z)`` and the message it sends towards ``x``. Which formula computes it
depends on the approximation.

[Belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation) integrates the
node function against the messages on the other interfaces:

```math
\mu_{f \to x}(x) = \int f(x, y, z)\, \mu_{y \to f}(y)\, \mu_{z \to f}(z) \, \mathrm{d}y\, \mathrm{d}z .
```

Its inputs are the messages ``\mu_{y \to f}`` and ``\mu_{z \to f}``, written `m[:y]` and `m[:z]`.

[Variational message passing](@extref MessagePassingRulesBase glossary-vmp) takes the expected
log-density under the marginals ``q(y)`` and ``q(z)``, the current posterior beliefs about the
neighbors:

```math
\mu_{f \to x}(x) \propto \exp \mathbb{E}_{q(y)\, q(z)}\big[\log f(x, y, z)\big] .
```

Its inputs are the marginals, written `q[:y]` and `q[:z]`. An expectation needs the whole
posterior of a neighbor, not only the message the neighbor sends.

[Structured variational message passing](@extref MessagePassingRulesBase glossary-structured-vmp)
mixes the two. When ``x`` and ``y`` stay joint and ``z`` is separate, the rule integrates over
``y`` with its message and takes the expectation over ``z`` with its marginal:

```math
\mu_{f \to x}(x) \propto \int \mu_{y \to f}(y) \exp \mathbb{E}_{q(z)}\big[\log f(x, y, z)\big] \, \mathrm{d}y .
```

Its inputs are `m[:y]` and `q[:z]`.

## How RxInfer finds a rule

RxInfer finds a rule from four things:

- the **node**, such as `NormalMeanPrecision` or the function `+`;
- the **target**, the interface the message is heading to;
- the node's **algorithm**, which is `DefaultAlgorithm()` for most nodes;
- the **inputs**: which messages and marginals the rule receives, and their types.

Among the rules with the same inputs, the types select one, as Julia's multiple dispatch selects
a method. [`which_message_update_rule`](@extref MessagePassingRulesBase.which_message_update_rule)
finds the rule for a call without running it:

```@example rules
which_message_update_rule(
    NormalMeanPrecision, :μ;
    m = (out = NormalMeanVariance(1.0, 2.0), τ = PointMass(4.0)),
)
```

The card shows the inputs the rule takes, where it is defined, and its body. With a normal message
on `out` and a known precision, the message towards the mean is a normal with the two variances
added.

[`rule_coverage`](@extref MessagePassingRulesBase.rule_coverage) tabulates every rule of a node:

```@example rules
MessagePassingRulesBase.rule_coverage(NormalMeanPrecision)
```

A row is a target, the average energy, or the joint marginal of a
[cluster](@extref MessagePassingRulesBase glossary-cluster). A column is an algorithm, and each
cell counts the rules. The six rules towards `μ` differ in their inputs: messages, marginals or a
mix, from a known value or from a distribution.

## Messages or marginals: the factorization decides

A rule does not choose whether it takes messages or marginals. The
[factorization](@extref MessagePassingRulesBase glossary-factorisation) of the posterior does,
which you state with [`@constraints`](@ref user-guide-constraints-specification). It splits each
node's interfaces into clusters. The rule towards a target then takes:

- the messages on the other interfaces of the target's own cluster;
- the marginals of the other clusters, a joint marginal for a cluster of several interfaces.

For a node `x ~ NormalMeanPrecision(μ, τ)`, with the interfaces `out`, `μ` and `τ`:

| factorization | clusters | the rule towards `out` takes | which is |
|:--|:--|:--|:--|
| `q(x, μ, τ)` | `(out, μ, τ)` | `m[:μ]`, `m[:τ]` | belief propagation |
| `q(x) q(μ) q(τ)` | `(out)`, `(μ)`, `(τ)` | `q[:μ]`, `q[:τ]` | [mean-field](@extref MessagePassingRulesBase glossary-mean-field) variational message passing |
| `q(x, μ) q(τ)` | `(out, μ)`, `(τ)` | `m[:μ]`, `q[:τ]` | structured variational message passing |

Under the mean-field factorization, the rule towards `out` reads the marginals. The card draws
marginal inputs as dashed arrows:

```@example rules
@call_message_update_rule(
    node = NormalMeanPrecision, target = :out,
    q = (μ = NormalMeanVariance(1.0, 2.0), τ = GammaShapeRate(2.0, 1.0)),
)
```

Under the structured factorization, the rule towards `τ` reads the joint marginal of the cluster
`(out, μ)`, written `q[:out, :μ]`. A call passes a joint marginal with the `clusters` keyword:

```@example rules
which_message_update_rule(
    NormalMeanPrecision, :τ;
    clusters = ((:out, :μ) => MvNormalMeanCovariance([1.0, 0.0], [1.0 0.5; 0.5 2.0]),),
)
```

Observed data and constants are known values, and RxInfer keeps each of them in a cluster of its
own. A stochastic node's rules therefore receive them as
[point-mass](@extref MessagePassingRulesBase glossary-point-mass) marginals, such as
`q[:out]::PointMass` for an observation. A point mass is the same distribution as a message or
as a marginal, and the rules return the same result for either.

A [deterministic node](@extref MessagePassingRulesBase glossary-deterministic-node), such as
`+` or `*`, relates its output to its inputs by a function. Its clusters are always its output
and the joint over its inputs, whatever the factorization, so its rules take messages.

## A node's algorithm

An [algorithm](@extref MessagePassingRulesBase glossary-algorithm) selects which rules a node
runs and carries their parameters. Most nodes run under `DefaultAlgorithm()`, where the
factorization alone decides the inputs. The node `*` runs under its own default,
[`MultiplicationSampling`](@extref StandardMessagePassingRules.MultiplicationSampling). Its
parameter is the number of samples that its rules for two uncertain factors draw:

```@example rules
MessagePassingRulesBase.rule_coverage(*)
```

A nonlinear function in a model is a [Delta node](@ref delta-node-manual), whose algorithm names
the approximation, such as `Linearization()` or `Unscented()`.
[Algorithm specification](@ref user-guide-algorithm-specification) shows how a model gives a node its
algorithm, with `where { algorithm = … }` or `@algorithm`.
[Algorithms and dependencies](@extref MessagePassingRulesBase Algorithms-and-dependencies)
describes the scheme in full.

## When no rule fits

When no rule takes the inputs a message needs, inference stops with a
[`RuleNotFoundError`](@extref MessagePassingRulesBase.RuleNotFoundError). The error names the
node, the target and the inputs, and explains, for every rule of that node and target, why the
rule does not fit. [Variational message passing by hand](@ref learning-vmp-by-hand) shows one,
and [Rule Not Found Error](@ref rule-not-found) lists the ways out: another factorization, a
functional form constraint, or a rule of your own.

## [Implementing a custom node](@id implementing-a-custom-node)

A model may need a node or a rule that no package defines.
[Creating your own custom nodes](@ref create-node) declares a node and writes its rules for
RxInfer. MessagePassingRulesBase's tutorial,
[Your first node](@extref MessagePassingRulesBase tutorial-first-node), writes the belief
propagation and variational rules of one node and checks them.
