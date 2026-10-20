# [Variational message passing by hand](@id learning-vmp-by-hand)

This page infers the mean and the precision of a normal distribution from independent
observations. Belief propagation has no closed form for this model, so you factorize the
posterior, run variational message passing by hand, and watch the free energy decrease. Then
[`infer`](@ref) runs the same updates. [Messages by hand](@ref learning-messages-by-hand) covers
belief propagation, which this page builds on.

The model has an unknown mean ``\mu`` and an unknown precision ``\tau``, the inverse of the
variance:

```math
\mu \sim \mathcal{N}(0, 100), \qquad \tau \sim \mathrm{Gamma}(1, 1), \qquad
y_i \sim \mathcal{N}(\mu, \tau^{-1}), \quad i = 1, \dots, 20 .
```

The Gamma distribution has shape ``1`` and rate ``1``. The data come from a normal with mean
``2`` and standard deviation ``0.5``:

```@example vmp-by-hand
using RxInfer, BayesBase, StableRNGs

y = 2.0 .+ 0.5 .* randn(StableRNG(42), 20)

@model function iid_normal(y)
    μ ~ NormalMeanVariance(0.0, 100.0)
    τ ~ Gamma(shape = 1.0, rate = 1.0)
    for i in eachindex(y)
        y[i] ~ NormalMeanPrecision(μ, τ)
    end
end
nothing # hide
```

## Belief propagation has no rule here

Belief propagation sends each likelihood node's message towards ``\mu`` as an integral over
``\tau``. With a Gamma message ``\mathrm{Gamma}(\tau \mid \alpha, \beta)`` on ``\tau``, the integral
is a Student's t density in ``\mu``:

```math
\int \mathcal{N}(y_i \mid \mu, \tau^{-1})\, \mathrm{Gamma}(\tau \mid \alpha, \beta) \, \mathrm{d}\tau
\;\propto\; \Big(1 + \frac{(y_i - \mu)^2}{2\beta}\Big)^{-(\alpha + 1/2)} .
```

The product of twenty such densities and a normal prior belongs to no family with a closed form,
and no rule computes it. The graph also has a loop: every message towards ``\mu`` needs the
message on ``\tau``, and that message needs the messages from the other observations, which need
``\mu``. Belief propagation on a loop needs an initial message to start. The
[initialization](@ref initialization) below gives one on ``\tau``, and inference stops at the
first rule it needs:

```@example vmp-by-hand
try
    infer(
        model = iid_normal(), data = (y = y,),
        initialization = @initialization(begin μ(τ) = GammaShapeRate(1.0, 1.0) end),
        disable_inference_error_hint = true,
    )
catch err
    showerror(stdout, err)
end
```

The [`RuleNotFoundError`](@extref MessagePassingRulesBase.RuleNotFoundError) names the node, the
target and the inputs, and explains why each rule of `NormalMeanPrecision` towards `μ` does not
fit. No rule takes a Gamma message `m[:τ]`. The observation arrives as `q[:out]`, a point-mass
marginal, because RxInfer keeps each observation in a
[cluster](@extref MessagePassingRulesBase glossary-cluster) of its own. The error suggests a
different factorization, which the next section introduces.

## A factorized posterior

Variational inference replaces the exact posterior with the closest distribution ``q`` from a
family you choose. It minimizes the free energy

```math
F[q] = \mathbb{E}_{q}\big[\log q(\mu, \tau) - \log p(y, \mu, \tau)\big] = -\log p(y) + \mathrm{KL}\big[q \,\|\, p(\cdot \mid y)\big],
```

which is never below ``-\log p(y)`` and reaches it at the exact posterior. The
[factorization](@extref MessagePassingRulesBase glossary-factorisation) chooses the family. The
[mean-field](@extref MessagePassingRulesBase glossary-mean-field) factorization
``q(\mu, \tau) = q(\mu)\, q(\tau)`` makes ``\mu`` and ``\tau`` independent under ``q``.

With ``q(\tau)`` fixed, the ``q(\mu)`` that minimizes ``F`` is

```math
q(\mu) \propto \exp \mathbb{E}_{q(\tau)}\big[\log p(y, \mu, \tau)\big]
       = \mathcal{N}(\mu \mid 0, 100) \prod_{i=1}^{20} \exp \mathbb{E}_{q(\tau)}\big[\log \mathcal{N}(y_i \mid \mu, \tau^{-1})\big] .
```

Each factor of the product is the message of one likelihood node towards ``\mu``. It is the
exponentiated expected log-density, a
[variational message passing](@extref MessagePassingRulesBase glossary-vmp) message, and it
reads the [marginal](@extref MessagePassingRulesBase glossary-marginal) ``q(\tau)`` instead of a
message. The expectation keeps only the mean of ``\tau``:

```math
\exp \mathbb{E}_{q(\tau)}\big[\log \mathcal{N}(y_i \mid \mu, \tau^{-1})\big]
\;\propto\; \exp\Big(-\frac{\mathbb{E}[\tau]}{2} (y_i - \mu)^2\Big)
\;\propto\; \mathcal{N}\big(\mu \mid y_i, \mathbb{E}[\tau]^{-1}\big) .
```

The rule takes marginals with the `q` keyword:

```@example vmp-by-hand
@call_message_update_rule(
    node = NormalMeanPrecision, target = :μ,
    q = (out = PointMass(y[1]), τ = GammaShapeRate(2.0, 1.0)),
)
```

The dashed arrows are marginal inputs. The message is a normal around the observation, with
precision ``\mathbb{E}[\tau] = 2``. The message towards ``\tau`` takes the expectation over
``q(\mu)`` instead, and is a Gamma with shape ``3/2`` and rate
``\big((y_i - \mathbb{E}[\mu])^2 + \mathrm{Var}[\mu]\big)/2``:

```@example vmp-by-hand
@call_message_update_rule(
    node = NormalMeanPrecision, target = :τ,
    q = (out = PointMass(y[1]), μ = NormalMeanVariance(2.0, 0.1)),
)
```

The Gamma prints with its scale, the inverse of the rate.

## Coordinate ascent by hand

The update of ``q(\mu)`` needs ``q(\tau)``, and the update of ``q(\tau)`` needs ``q(\mu)``.
Coordinate ascent alternates them, from an initial guess for ``q(\tau)``. Each update multiplies
the prior's message and the twenty likelihood messages:

```@example vmp-by-hand
prior_μ = getresult(@call_message_update_rule(node = NormalMeanVariance, target = :out, m = (μ = PointMass(0.0), v = PointMass(100.0))))
prior_τ = getresult(@call_message_update_rule(node = GammaShapeRate, target = :out, m = (α = PointMass(1.0), β = PointMass(1.0))))

multiply(first, messages) = foldl((left, right) -> prod(ClosedProd(), left, right), messages; init = first)

function update_μ(q_τ)
    messages = [getresult(@call_message_update_rule(node = NormalMeanPrecision, target = :μ, q = (out = PointMass(yi), τ = q_τ))) for yi in y]
    return multiply(prior_μ, messages)
end

function update_τ(q_μ)
    messages = [getresult(@call_message_update_rule(node = NormalMeanPrecision, target = :τ, q = (out = PointMass(yi), μ = q_μ))) for yi in y]
    return multiply(prior_τ, messages)
end
nothing # hide
```

`Gamma(shape = 1.0, rate = 1.0)` in the model creates a `GammaShapeRate` node, whose interfaces
are `out`, `α` and `β`.

The free energy of this factorization is a sum over the nodes and the variables. Each node
contributes its [average energy](@extref MessagePassingRulesBase glossary-average-energy),
``U_f = -\mathbb{E}_q[\log f]``, and each latent variable subtracts the entropy of its marginal:

```math
F = U_{\mu} + U_{\tau} + \sum_{i=1}^{20} U_{i} - \mathrm{H}[q(\mu)] - \mathrm{H}[q(\tau)] .
```

[`@call_average_energy`](@extref MessagePassingRulesBase.@call_average_energy) computes a node's
average energy from the marginals of its interfaces:

```@example vmp-by-hand
@call_average_energy(
    node = NormalMeanPrecision,
    q = (out = PointMass(y[1]), μ = NormalMeanVariance(2.0, 0.1), τ = GammaShapeRate(2.0, 1.0)),
)
```

The free energy adds the average energies of the two priors and the twenty likelihoods:

```@example vmp-by-hand
function free_energy(q_μ, q_τ)
    U = getresult(@call_average_energy(node = NormalMeanVariance, q = (out = q_μ, μ = PointMass(0.0), v = PointMass(100.0))))
    U += getresult(@call_average_energy(node = GammaShapeRate, q = (out = q_τ, α = PointMass(1.0), β = PointMass(1.0))))
    U += sum(getresult(@call_average_energy(node = NormalMeanPrecision, q = (out = PointMass(yi), μ = q_μ, τ = q_τ))) for yi in y)
    return U - entropy(q_μ) - entropy(q_τ)
end

function coordinate_ascent(q_τ; iterations)
    local q_μ
    free_energies = Float64[]
    for _ in 1:iterations
        q_μ = update_μ(q_τ)
        q_τ = update_τ(q_μ)
        push!(free_energies, free_energy(q_μ, q_τ))
    end
    return q_μ, q_τ, free_energies
end

q_μ, q_τ, free_energies = coordinate_ascent(GammaShapeRate(1.0, 1.0); iterations = 10)
free_energies
```

The free energy decreases until it converges, after about five iterations. Compare the
posteriors with the data:

```@example vmp-by-hand
(mean_μ = mean(q_μ), sample_mean = mean(y), mean_τ = mean(q_τ), sample_precision = 1 / var(y))
```

The posterior of ``\mu`` centers on the sample mean. The posterior of ``\tau`` lies below the
sample precision. The prior adds its rate, ``1``, to the rate of ``q(\tau)``, next to about
``2.6`` from the data, and so pulls ``\tau`` down.

## The same inference in RxInfer

[`@constraints`](@ref user-guide-constraints-specification) states the factorization, and
[`@initialization`](@ref initialization) the initial guess for ``q(\tau)``:

```@example vmp-by-hand
result = infer(
    model = iid_normal(),
    data = (y = y,),
    constraints = @constraints(begin
        q(μ, τ) = q(μ)q(τ)
    end),
    initialization = @initialization(begin
        q(τ) = GammaShapeRate(1.0, 1.0)
    end),
    iterations = 10,
    free_energy = true,
)
```

RxInfer runs the same rules in the same order, so the free energies and the posteriors agree
with yours:

```@example vmp-by-hand
(
    free_energy = result.free_energy ≈ free_energies,
    μ = mean_var(result.posteriors[:μ][end]) .≈ mean_var(q_μ),
    τ = mean_var(result.posteriors[:τ][end]) .≈ mean_var(q_τ),
)
```

## Structured factorizations

The mean-field factorization splits every variable from every other. A structured factorization
keeps some variables together. In a state-space model with an unknown precision ``\tau``,
``q(x_{1:T}, \tau) = q(x_{1:T})\, q(\tau)`` keeps the states joint. The transition nodes then
see the states in one cluster and ``\tau`` in another. Their rule towards the previous state,
`μ`, takes the message from the next state, `m[:out]`, and the marginal `q[:τ]`:

```@example vmp-by-hand
which_message_update_rule(
    NormalMeanPrecision, :μ;
    m = (out = NormalMeanVariance(0.0, 1.0),), q = (τ = GammaShapeRate(2.0, 1.0),),
)
```

[`which_message_update_rule`](@extref MessagePassingRulesBase.which_message_update_rule) finds
the rule a call would run, without running it. A structured factorization keeps the correlations
that mean field discards, at the cost of larger clusters. [Understanding rules](@ref what-is-a-rule)
tabulates which inputs each factorization gives a rule, and
[Constraints specification](@ref user-guide-constraints-specification) writes factorizations for
larger models.
