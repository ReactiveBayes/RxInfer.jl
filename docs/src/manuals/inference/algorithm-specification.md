# [Algorithm specification](@id user-guide-algorithm-specification)

Every rule of a node belongs to an [algorithm](@extref MessagePassingRulesBase glossary-algorithm).
The algorithm selects which rules run and carries their parameters. Most nodes run under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), and a model says nothing
about them. Some nodes need a choice from you:

- the `AR` node, for autoregressive processes, needs the order of the process and whether it is
  univariate or multivariate: [`ARVMP`](@extref AutoregressiveMessagePassingRules.ARVMP);
- the `GCV` node, the [Gaussian controlled variance](https://ieeexplore.ieee.org/document/9173980),
  computes its messages with a cubature whose number of points you choose:
  [`GCVApproximation`](@extref GCVMessagePassingRules.GCVApproximation);
- a deterministic node, such as `y := f(x)`, needs an approximation method:
  [`DeltaApproximation`](@extref DeltaMessagePassingRules.DeltaApproximation), see
  [Deterministic nodes](@ref delta-node-manual).

You give a node its algorithm in the model, with `where { algorithm = ... }`, or for the whole
model with the [`@algorithm`](@ref) macro and the `algorithm` keyword of [`infer`](@ref). The
`meta` keyword, the `@meta` macro and `where { meta = ... }` still work, with a deprecation
warning, as the old names of these three.

## An algorithm in the model

The `AR` node lives in the `AutoregressiveMessagePassingRules` package, which you load next to
`RxInfer`. This model observes an autoregressive process of order 2 through noise:

```@example algorithm-specification
using RxInfer, AutoregressiveMessagePassingRules, StableRNGs

@model function ar_model(y, order)
    γ  ~ Gamma(shape = 1.0, rate = 1.0)
    θ  ~ MvNormal(mean = zeros(order), precision = diageye(order))
    x0 ~ MvNormal(mean = zeros(order), precision = diageye(order))
    c = [1.0; zeros(order - 1)]
    x_prev = x0
    for i in eachindex(y)
        x[i] ~ AR(x_prev, θ, γ) where { algorithm = ARVMP(Multivariate, order, ARsafe()) }
        y[i] ~ Normal(mean = dot(c, x[i]), precision = 10.0)
        x_prev = x[i]
    end
end
```

`ARVMP(Multivariate, order, ARsafe())` is the algorithm of each `AR` node: a multivariate state
of length `order`, with the numerically safe computation of the joint over the states. The
node's rules are variational, so the model needs a factorization and initial marginals:

```@example algorithm-specification
θ_real = [0.5, -0.3]
series = zeros(102)
rng    = StableRNG(42)
for t in 3:102
    series[t] = θ_real[1] * series[t - 1] + θ_real[2] * series[t - 2] + randn(rng)
end
dataset = series[3:end] .+ randn(rng, 100) ./ sqrt(10.0)

ar_constraints = @constraints begin
    q(x, x0, γ, θ) = q(x, x0)q(γ)q(θ)
end

ar_initialization = @initialization begin
    q(γ) = GammaShapeRate(1.0, 1.0)
    q(θ) = MvNormalMeanPrecision(zeros(2), diageye(2))
end

result = infer(
    model          = ar_model(order = 2),
    data           = (y = dataset,),
    constraints    = ar_constraints,
    initialization = ar_initialization,
    iterations     = 20,
)

mean(result.posteriors[:θ][end])
```

The posterior mean of the coefficients is close to `θ_real`. Without an algorithm, no rule of
`AR` applies, and inference stops with an error that says so:

```@example algorithm-specification
@model function ar_model_without_algorithm(y, order)
    γ  ~ Gamma(shape = 1.0, rate = 1.0)
    θ  ~ MvNormal(mean = zeros(order), precision = diageye(order))
    x0 ~ MvNormal(mean = zeros(order), precision = diageye(order))
    c = [1.0; zeros(order - 1)]
    x_prev = x0
    for i in eachindex(y)
        x[i] ~ AR(x_prev, θ, γ)
        y[i] ~ Normal(mean = dot(c, x[i]), precision = 10.0)
        x_prev = x[i]
    end
end

try
    infer(
        model          = ar_model_without_algorithm(order = 2),
        data           = (y = dataset,),
        constraints    = ar_constraints,
        initialization = ar_initialization,
        iterations     = 20,
        disable_inference_error_hint = true, #hide
    )
catch err
    showerror(stdout, err)
end
```

## The `@algorithm` macro

`@algorithm` gives algorithms to nodes from outside the model. Each line selects nodes by their
function and, optionally, by the variables they connect, and gives the algorithm after `->`:

```@example algorithm-specification
ar_algorithm = @algorithm begin
    AR() -> ARVMP(Multivariate, 2, ARsafe())
end

result = infer(
    model          = ar_model_without_algorithm(order = 2),
    data           = (y = dataset,),
    constraints    = ar_constraints,
    initialization = ar_initialization,
    algorithm      = ar_algorithm,
    iterations     = 20,
)

mean(result.posteriors[:θ][end])
```

`AR()` selects every `AR` node of the model. The result is the same as with
`where { algorithm = ... }`. An algorithm given in the model with `where` takes precedence over
one from `@algorithm`.

`@algorithm` also defines a function that returns a specification, so that the algorithm can
depend on arguments:

```@example algorithm-specification
@algorithm function ar_algorithm_of_order(order)
    AR() -> ARVMP(Multivariate, order, ARsafe())
end

ar_algorithm = ar_algorithm_of_order(2)
nothing # hide
```

```@docs
RxInfer.@algorithm
```

## Selecting nodes by their variables

The `GCV` node lives in the `GCVMessagePassingRules` package. It models an observation whose
log-variance is linear in a latent variable, ``y \sim \mathcal{N}(x, \exp(\kappa z + \omega))``:

```@example algorithm-specification
using GCVMessagePassingRules

@model function volatility_model(y, κ, ω)
    x ~ Normal(mean = 0.0, variance = 10.0)
    z ~ Normal(mean = 0.0, variance = 1.0)
    for i in eachindex(y)
        y[i] ~ GCV(x, z, κ, ω)
    end
end
```

Its default algorithm, `GCVApproximation()`, uses a Gauss–Hermite cubature with 20 points. The
entry `GCV(x, z) -> ...` selects the `GCV` nodes connected to both `x` and `z`, and gives them a
cubature with 32 points:

```@example algorithm-specification
volatility_algorithm = @algorithm begin
    GCV(x, z) -> GCVApproximation(method = GaussHermiteCubature(32))
end

volatility_initialization = @initialization begin
    q(x) = NormalMeanVariance(0.0, 10.0)
    q(z) = NormalMeanVariance(0.0, 1.0)
end

volatility_data = 1.0 .+ sqrt(exp(0.5)) .* randn(StableRNG(1), 200)

result = infer(
    model          = volatility_model(κ = 1.0, ω = 0.0),
    data           = (y = volatility_data,),
    constraints    = MeanField(),
    initialization = volatility_initialization,
    algorithm      = volatility_algorithm,
    iterations     = 20,
)

(x = mean(result.posteriors[:x][end]), z = mean(result.posteriors[:z][end]))
```

The data have mean 1 and log-variance 0.5, which the posteriors of `x` and `z` recover. A
specification holds as many entries as you need, one per line:

```@example algorithm-specification
combined_algorithm = @algorithm begin
    GCV(x, z) -> GCVApproximation(method = GaussHermiteCubature(32))
    AR() -> ARVMP(Multivariate, 2, ARsafe())
end
nothing # hide
```

## Algorithms for nodes in submodels

As in `@constraints`, a `for meta in submodel` block gives algorithms to the nodes of a submodel:

```@example algorithm-specification
@model function noisy_ar_step(y, x_next, x_prev, θ, γ, order)
    x_next ~ AR(x_prev, θ, γ)
    y ~ Normal(mean = dot([1.0; zeros(order - 1)], x_next), precision = 10.0)
end

@model function ar_model_with_submodel(y, order)
    γ  ~ Gamma(shape = 1.0, rate = 1.0)
    θ  ~ MvNormal(mean = zeros(order), precision = diageye(order))
    x0 ~ MvNormal(mean = zeros(order), precision = diageye(order))
    x_prev = x0
    for i in eachindex(y)
        x[i] ~ noisy_ar_step(y = y[i], x_prev = x_prev, θ = θ, γ = γ, order = order)
        x_prev = x[i]
    end
end

submodel_algorithm = @algorithm begin
    for meta in noisy_ar_step
        AR() -> ARVMP(Multivariate, 2, ARsafe())
    end
end

result = infer(
    model          = ar_model_with_submodel(order = 2),
    data           = (y = dataset,),
    constraints    = ar_constraints,
    initialization = ar_initialization,
    algorithm      = submodel_algorithm,
    iterations     = 20,
)

mean(result.posteriors[:θ][end])
```

The block is written `for meta in …` in both macros. The entries inside it apply only to the
nodes of `noisy_ar_step`, so the same node function can run under different algorithms in
different submodels.

## [A custom algorithm](@id user-guide-algorithm-specification-custom)

An algorithm is a type, and its fields are the parameters of its rules. You can define one for a
node you did not write, and give it rules of your own. This section tempers the likelihood of a
Gaussian observation: it raises ``\mathcal{N}(y \mid x, v)`` to a power ``\beta \in (0, 1]``,
which widens its message towards the mean,

```math
\mathcal{N}(y \mid x, v)^{\beta} \propto \mathcal{N}(x \mid y, v / \beta).
```

A subtype of [`DefaultAlgorithmExtension`](@extref MessagePassingRulesBase.DefaultAlgorithmExtension)
overrides some rules of the default algorithm and inherits the others:

```@example custom-algorithm
using RxInfer

struct Tempered{T} <: DefaultAlgorithmExtension
    β::T
end

@define_message_update_rule(
    node = NormalMeanVariance,
    target = :μ,
    algorithm = Tempered,
    args = (q[:out]::PointMass, q[:v]::PointMass),
    body = (algo, args) -> NormalMeanVariance(mean(args.q[:out]), mean(args.q[:v]) / algo.β),
)

@call_message_update_rule(
    node = NormalMeanVariance, target = :μ,
    q = (out = PointMass(4.0), v = PointMass(2.0)),
    algorithm = Tempered(0.5),
)
```

The body names the `algo` slot before `args`, and reads ``\beta`` from it. The rule takes
marginals because `y` is observed and `v` is a constant: each is a cluster of its own. The
[tutorial on algorithms](@extref MessagePassingRulesBase tutorial-algorithm) of
MessagePassingRulesBase covers algorithms that stand alone and declare what their rules take.

`@algorithm` gives the new algorithm to the node connected to `y`:

```@example custom-algorithm
@model function gaussian_model(y)
    x ~ NormalMeanVariance(2.5, 0.5)
    y ~ NormalMeanVariance(x, 2.0)
end

tempered = @algorithm begin
    NormalMeanVariance(y) -> Tempered(0.5)
end

result          = infer(model = gaussian_model(), data = (y = 4.0,))
result_tempered = infer(model = gaussian_model(), data = (y = 4.0,), algorithm = tempered)

(exact = mean_var(result.posteriors[:x]), tempered = mean_var(result_tempered.posteriors[:x]))
```

The tempered posterior stays closer to the prior mean, 2.5, and has a larger variance. Its
closed form is the product of the prior with ``\mathcal{N}(x \mid 4, 2 / 0.5)``:

```@example custom-algorithm
mean_var(prod(PreserveTypeProd(Distribution), NormalMeanVariance(2.5, 0.5), NormalMeanVariance(4.0, 4.0)))
```

The prior's node, `x ~ NormalMeanVariance(2.5, 0.5)`, keeps the default algorithm, because the
entry selects only the node connected to `y`.
