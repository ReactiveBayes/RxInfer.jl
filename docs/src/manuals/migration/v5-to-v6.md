# [Migration Guide from version 5.x to 6.x](@id migration-5-to-6)

This guide explains how to migrate a model from `RxInfer` 5.x to 6.x. RxInfer 6.x runs on
`ReactiveMP` v7, which moves every factor node and its rules out of the engine into rule
packages. The main breaking changes for a model author are:

1. **Some nodes need their package loaded.** RxInfer loads the standard nodes and the Delta node; the others have packages of their own.
2. **A node's meta is its algorithm**: `@meta` becomes `@algorithm` and `meta =` becomes `algorithm =`, and several meta types are renamed.
3. **`where { dependencies = … }` is removed**: a node declares what its rules read, and you set initial messages with `@initialization`.
4. **Log scales are part of a message**: `annotations = LogScaleAnnotations()` becomes `logscales = true`.
5. **Nodes and rules are declared with new macros**: `@node`, `@rule`, `@marginalrule`, `@average_energy` and `@call_rule` are replaced.

The sections below follow the order in which you meet these changes when you run an old script.
A block that starts with `# v6` shows the RxInfer 5.x code, which ran on ReactiveMP v6; the block
after it shows the 6.x code and runs when this documentation is built.

## Loading the node packages

RxInfer loads and re-exports `ReactiveMP`,
[`MessagePassingRulesBase`](@extref MessagePassingRulesBase MessagePassingRulesBase),
[`StandardMessagePassingRules`](@extref StandardMessagePassingRules StandardMessagePassingRules),
[`DeltaMessagePassingRules`](@extref DeltaMessagePassingRules DeltaMessagePassingRules) and
[`MessagePassingRulesApproximations`](@extref MessagePassingRulesApproximations MessagePassingRulesApproximations).
These cover the distributions, the arithmetic and logic nodes, the mixtures and deterministic
transformations with `:=`. Every other node lives in a package that you load next to RxInfer:

| Node | Package |
|:-----|:--------|
| `AR`, `ConjugateAR` | [`AutoregressiveMessagePassingRules`](@extref AutoregressiveMessagePassingRules AutoregressiveMessagePassingRules) |
| `GCV` | [`GCVMessagePassingRules`](@extref GCVMessagePassingRules GCVMessagePassingRules) |
| `Probit` | [`ProbitMessagePassingRules`](@extref ProbitMessagePassingRules ProbitMessagePassingRules) |
| `SoftDot` | [`SoftDotMessagePassingRules`](@extref SoftDotMessagePassingRules SoftDotMessagePassingRules) |
| `ContinuousTransition` | [`ContinuousTransitionMessagePassingRules`](@extref ContinuousTransitionMessagePassingRules ContinuousTransitionMessagePassingRules) |
| `BinomialPolya`, `MultinomialPolya` | [`PolyaMessagePassingRules`](@extref PolyaMessagePassingRules PolyaMessagePassingRules) (GPL-3 licensed) |
| `BIFM`, `BIFMHelper` | [`BIFMMessagePassingRules`](@extref BIFMMessagePassingRules BIFMMessagePassingRules) |
| `Flow` | [`FlowMessagePassingRules`](@extref FlowMessagePassingRules FlowMessagePassingRules) |
| `DiscreteTransition` | [`DiscreteTransitionMessagePassingRules`](@extref DiscreteTransitionMessagePassingRules DiscreteTransitionMessagePassingRules) |
| `GaussianCoupling` | [`GaussianCouplingMessagePassingRules`](@extref GaussianCouplingMessagePassingRules GaussianCouplingMessagePassingRules) |

Each package is an ordinary Julia package: install it with `] add AutoregressiveMessagePassingRules`
and load it with `using`. Loading it is enough for the engine to find the node's rules. A model
with an autoregressive state reads as follows:

```@example migration-packages
using RxInfer, AutoregressiveMessagePassingRules

@model function ar_model(y)
    θ ~ Normal(mean = 0.0, precision = 1.0)
    γ ~ Gamma(shape = 1.0, rate = 1.0)
    x_prev ~ Normal(mean = 0.0, precision = 1.0)
    for i in eachindex(y)
        x[i] ~ AR(x_prev, θ, γ) where { algorithm = ARVMP(Univariate, 1, ARsafe()) }
        y[i] ~ Normal(mean = x[i], precision = 10.0)
        x_prev = x[i]
    end
end

result = infer(
    model = ar_model(),
    data = (y = [0.5, 0.4, 0.3, 0.25, 0.2],),
    constraints = @constraints(begin
        q(x_prev, x, θ, γ) = q(x_prev, x)q(θ)q(γ)
    end),
    initialization = @initialization(begin
        q(θ) = NormalMeanPrecision(0.0, 1.0)
        q(γ) = GammaShapeRate(1.0, 1.0)
    end),
    iterations = 10,
)

mean(result.posteriors[:θ][end])
```

Without `using AutoregressiveMessagePassingRules`, the name `AR` is undefined, and the error
says which package to load (see [Hints for removed names](@ref migration-5-to-6-hints)).

## [A node's meta is its algorithm](@id migration-5-to-6-algorithm)

In ReactiveMP v7, the setting that chooses how a node computes its messages is the node's
[algorithm](@extref MessagePassingRulesBase glossary-algorithm). RxInfer follows the new name, with the [`@algorithm`](@ref) macro:

| RxInfer 5.x | RxInfer 6.x |
|:------------|:------------|
| `@meta begin … end` | `@algorithm begin … end` |
| `infer(…; meta = …)` | `infer(…; algorithm = …)` |
| `x ~ Node(…) where { meta = … }` | `x ~ Node(…) where { algorithm = … }` |
| `DeltaMeta(method = …, inverse = …)` | [`DeltaApproximation`](@extref DeltaMessagePassingRules.DeltaApproximation)`(method = …, inverse = …)` |
| `ARMeta(form, order, stype)` | [`ARVMP`](@extref AutoregressiveMessagePassingRules.ARVMP)`(form, order, stype)` |
| `GCVMetadata(method)` | [`GCVApproximation`](@extref GCVMessagePassingRules.GCVApproximation)`(method = method)` |
| `CTMeta(f)`, `ContinuousTransitionMeta(f)` | [`CTVMP`](@extref ContinuousTransitionMessagePassingRules.CTVMP)`(f)` |
| `FlowMeta(model, approximation)` | [`FlowApproximation`](@extref FlowMessagePassingRules.FlowApproximation)`(model; method = approximation)` |

`meta`, `@meta` and `where { meta = … }` still work in 6.x and print a deprecation warning. The
renamed meta types are gone. ReactiveMP's [node packages table](@extref ReactiveMP migration-v6-to-v7-node-packages)
lists the algorithms of the other nodes.

```julia
# v6
result = infer(
    model = square_model(),
    data = (y = 2.0,),
    meta = @meta(begin
        f() -> DeltaMeta(method = Linearization())
    end),
)
```

```@example migration-algorithm
using RxInfer

f(x) = x^2

@model function square_model(y)
    x ~ Normal(mean = 1.0, variance = 1.0)
    z := f(x)
    y ~ Normal(mean = z, variance = 0.1)
end

result = infer(
    model = square_model(),
    data = (y = 2.0,),
    algorithm = @algorithm(begin
        f() -> DeltaApproximation(method = Linearization())
    end),
)

result.posteriors[:x]
```

A single node takes its algorithm in the model, with the same `where` clause that held its meta:

```@example migration-algorithm
@model function square_model_unscented(y)
    x ~ Normal(mean = 1.0, variance = 1.0)
    z := f(x) where { algorithm = DeltaApproximation(method = Unscented()) }
    y ~ Normal(mean = z, variance = 0.1)
end

result = infer(model = square_model_unscented(), data = (y = 2.0,))

result.posteriors[:x]
```

A Delta node also accepts its approximation method alone, `f() -> Linearization()`, as in 5.x.
See [Algorithm specification](@ref user-guide-algorithm-specification) and
[Deterministic nodes](@ref delta-node-manual) for the details.

## `where { dependencies = … }` is removed

In 5.x a model could change which inputs a node's rules read, and start a message on the node's
own edge, with `where { dependencies = RequireMessageFunctionalDependencies(…) }`. In 6.x a node
declares what its rules read ([dependencies](@extref MessagePassingRulesBase glossary-dependencies)),
and you set an [initial message](@extref MessagePassingRulesBase glossary-initial-message) with
[`@initialization`](@ref). A model that still uses the clause fails with an error that says so.

```julia
# v6
@model function binomial_model(X, n, y)
    β ~ MvNormalWeightedMeanPrecision(zeros(2), diageye(2))
    for i in eachindex(y)
        y[i] ~ BinomialPolya(X[i], n[i], β) where {
            dependencies = RequireMessageFunctionalDependencies(β = MvNormalWeightedMeanPrecision(zeros(2), diageye(2)))
        }
    end
end
```

```@example migration-dependencies
using RxInfer, PolyaMessagePassingRules, StableRNGs

@model function binomial_model(X, n, y)
    β ~ MvNormalWeightedMeanPrecision(zeros(2), diageye(2))
    for i in eachindex(y)
        y[i] ~ BinomialPolya(X[i], n[i], β)
    end
end

rng = StableRNG(42)
X = [randn(rng, 2) for _ in 1:30]
n = fill(10, 30)
y = [rand(rng, Binomial(10, 1 / (1 + exp(x[2] - x[1])))) for x in X]

result = infer(
    model = binomial_model(),
    data = (X = X, n = n, y = y),
    initialization = @initialization(begin
        μ(β) = MvNormalWeightedMeanPrecision(zeros(2), diageye(2))
    end),
    iterations = 5,
)

mean(result.posteriors[:β][end])
```

`BinomialPolya` declares that its rule towards `β` reads the message on `β`'s own edge, so the
model only initializes that message. A per-node option, `where { initial_messages = … }`, is being
added as the counterpart of the removed clause for a single node; until it is released, use
`@initialization`.

## Log scales

A [log scale](@extref MessagePassingRulesBase glossary-log-scale) is part of every message and
marginal in ReactiveMP v7, not an annotation. The `logscales` option replaces
`LogScaleAnnotations`, and you read a posterior's log scale with `getlogscale` directly.

```julia
# v6
result = infer(model = coin_model(), data = (y = y,), annotations = LogScaleAnnotations())
getlogscale(getannotations(result.posteriors[:θ]))
```

```@example migration-logscales
using RxInfer

@model function coin_model(y)
    θ ~ Beta(1.0, 1.0)
    y .~ Bernoulli(θ)
end

result = infer(model = coin_model(), data = (y = [1.0, 0.0, 1.0],), logscales = true)

getlogscale(result.posteriors[:θ])
```

In a model that belief propagation solves exactly, a posterior's log scale is the log evidence
of the data: here `log(1/12)`. `options = (logscales = true,)` is the same option.

## Custom nodes and rules in scripts

A script that declares its own node uses the macros of
[`MessagePassingRulesBase`](@extref MessagePassingRulesBase MessagePassingRulesBase), which RxInfer
re-exports. Each old macro has one counterpart:

| RxInfer 5.x | RxInfer 6.x |
|:------------|:------------|
| `@node` | [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node) |
| `@rule` | [`@define_message_update_rule`](@extref MessagePassingRulesBase.@define_message_update_rule) |
| `@marginalrule` | [`@define_marginal_update_rule`](@extref MessagePassingRulesBase.@define_marginal_update_rule) |
| `@average_energy` | [`@define_average_energy`](@extref MessagePassingRulesBase.@define_average_energy) |
| `@call_rule` | [`@call_message_update_rule`](@extref MessagePassingRulesBase.@call_message_update_rule) |
| `@logscale v` in a rule's body | the rule's `logscale = v` keyword |
| `Marginalisation`, `MomentMatching` | removed: a rule belongs to its node's default algorithm unless it names one with `algorithm = …` |

The pairs below port a Bernoulli-like node. A node is declared keyword by keyword:

```julia
# v6
struct MyBernoulli end
@node MyBernoulli Stochastic [out, p]
```

```@example migration-custom
using RxInfer

struct MyBernoulli end

@define_factor_node(node = MyBernoulli, type = Stochastic, interfaces = [:out, :p])
```

A message rule names its node, its target and its inputs as keywords. The inputs `m_x` and `q_x`
become `m[:x]` and `q[:x]`, a joint marginal `q_x_y` becomes `q[:x, :y]`, and the body is a
function of `args`. The log scale is a keyword of the rule:

```julia
# v6
@rule MyBernoulli(:p, Marginalisation) (q_out::PointMass,) = begin
    @logscale -log(2)
    return Beta(1 + mean(q_out), 2 - mean(q_out))
end
```

```@example migration-custom
@define_message_update_rule(
    node = MyBernoulli, target = :p,
    args = (q[:out]::PointMass,),
    logscale = -log(2),
    body = (args) -> Beta(1 + mean(args.q[:out]), 2 - mean(args.q[:out])),
)
```

A marginal rule's target is the tuple of its members, `(:out, :p)` in place of `:out_p`. A result
that factorizes into independent blocks is a
[`FactorizedCluster`](@extref MessagePassingRulesBase.FactorizedCluster) in place of a `NamedTuple`:

```julia
# v6
@marginalrule MyBernoulli(:out_p) (m_out::PointMass, m_p::Beta) = begin
    return (out = m_out, p = prod(ClosedProd(), Beta(1 + mean(m_out), 2 - mean(m_out)), m_p))
end
```

```@example migration-custom
@define_marginal_update_rule(
    node = MyBernoulli, target = (:out, :p),
    args = (m[:out]::PointMass, m[:p]::Beta),
    body = (args) -> FactorizedCluster(
        (:out,) => args.m[:out],
        (:p,) => prod(ClosedProd(), Beta(1 + mean(args.m[:out]), 2 - mean(args.m[:out])), args.m[:p]),
    ),
)
```

An average energy takes the same argument syntax as a rule:

```julia
# v6
@average_energy MyBernoulli (q_out::PointMass, q_p::Beta) = begin
    return -(mean(q_out) * mean(log, q_p) + (1 - mean(q_out)) * mean(mirrorlog, q_p))
end
```

```@example migration-custom
@define_average_energy(
    node = MyBernoulli,
    args = (q[:out]::PointMass, q[:p]::Beta),
    body = (args) -> -(mean(args.q[:out]) * mean(log, args.q[:p]) + (1 - mean(args.q[:out])) * mean(mirrorlog, args.q[:p])),
)
```

The node works in a model as before:

```@example migration-custom
@model function my_coin_model(y)
    p ~ Beta(1.0, 1.0)
    y .~ MyBernoulli(p)
end

result = infer(model = my_coin_model(), data = (y = [1.0, 0.0, 1.0],), free_energy = true)

result.posteriors[:p], result.free_energy
```

A rule reads the services the engine provides, such as the random number generator, from its
context; a node with a variable number of edges declares an interface group. ReactiveMP's
[porting guide for rule authors](@extref ReactiveMP migration-v6-to-v7) covers these, and
[Creating your own custom nodes](@ref create-node) walks through a node from the start.

## Calling rules by hand

`@call_rule` becomes [`@call_message_update_rule`](@extref MessagePassingRulesBase.@call_message_update_rule).
It takes keywords: the node, the target, and the messages `m` and marginals `q` by interface
name. The interfaces of `+` are `out`, `in1` and `in2`, and
[`MessagePassingRulesBase.nodespec`](@extref) lists any node's:

```@example migration-call
using RxInfer

MessagePassingRulesBase.nodespec(+)
```

```julia
# v6
@call_rule typeof(+)(:out, Marginalisation) (m_in1 = NormalMeanVariance(1.0, 1.0), m_in2 = NormalMeanVariance(2.0, 1.0))
```

```@example migration-call
result = @call_message_update_rule(
    node = +, target = :out,
    m = (in1 = NormalMeanVariance(1.0, 1.0), in2 = NormalMeanVariance(2.0, 1.0)),
)
```

The call returns a [`RuleResult`](@extref MessagePassingRulesBase.RuleResult), which renders as a
card: the node, the inputs it read, the message it computed, its log scale and the rule that ran.
[`getresult`](@extref MessagePassingRulesBase.getresult) returns the message itself, the value
`@call_rule` returned:

```@example migration-call
getresult(result)
```

The interfaces of `*` are `out`, `A` and `in`. A known gain is a
[point mass](@extref MessagePassingRulesBase glossary-point-mass) on `A`:

```julia
# v6
@call_rule typeof(*)(:out, Marginalisation) (m_A = PointMass(2.0), m_in = NormalMeanVariance(1.0, 1.0))
```

```@example migration-call
getresult(@call_message_update_rule(
    node = *, target = :out,
    m = (A = PointMass(2.0), in = NormalMeanVariance(1.0, 1.0)),
))
```

A `meta = …` argument of `@call_rule` becomes `algorithm = …`.

## The rule-not-found report

When no rule matches, inference stops with a
[`RuleNotFoundError`](@extref MessagePassingRulesBase.RuleNotFoundError) that explains itself.
Belief propagation has no rule for the mean of a normal with an unknown precision:

```@example migration-not-found
using RxInfer

@model function normal_gamma_model(y)
    μ ~ Normal(mean = 0.0, variance = 1.0)
    τ ~ Gamma(shape = 1.0, rate = 1.0)
    y ~ Normal(mean = μ, precision = τ)
end

try
    infer(model = normal_gamma_model(), data = (y = 1.0,), disable_inference_error_hint = true)
catch err
    showerror(stdout, err)
end
```

The first line names the node, the target, the algorithm and the inputs the engine offered. The
`what to try` line suggests a fix, and each near miss marks which of a rule's inputs match. See
[Rule Not Found Error](@ref rule-not-found) for how to read and resolve it.

## New inference options

Two options of `infer`'s `options` argument are new in 6.x.

- `diagnostics` takes a [`ReactiveMP.EngineDiagnostics`](@extref), the engine's audits of the rules a model runs: `check_everything_pure` rejects a rule that declares itself impure, `check_everything_inplace` warns about a rule without an in-place form, and `checked_buffers` fills recycled memory with `NaN` before each reuse.
- `context` gives the rules their [services](@extref MessagePassingRulesBase glossary-service), a `NamedTuple` merged over the engine's defaults: `rng`, the random number generator that sampling rules draw from, and `matrix_correction`, the correction rules apply to the matrices they build.

```@example migration-logscales
using StableRNGs

result = infer(
    model = coin_model(),
    data = (y = [1.0, 0.0, 1.0],),
    options = (
        diagnostics = ReactiveMP.EngineDiagnostics(check_everything_pure = true),
        context = (rng = StableRNG(42),),
    ),
)

result.posteriors[:θ]
```

The `rulefallback` option, `options = (rulefallback = NodeFunctionRuleFallback(),)`, is unchanged:
it gives a message only where no rule matches.

## [Hints for removed names](@id migration-5-to-6-hints)

A name that ReactiveMP v7 removed or moved into a node package raises an `UndefVarError` with a
hint that names its replacement:

```@example migration-hints
using RxInfer

try
    eval(:(ARMeta(1)))
catch err
    showerror(stdout, err)
end
```

The same hint appears for `@node`, `@rule`, `@call_rule`, `Marginalisation`, the renamed meta
types, `LogScaleAnnotations`, the functional dependency types, and every node of a package that
is not loaded.
