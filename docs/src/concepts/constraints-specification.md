# [Constraints Specification](@id concepts-constraints-specification)

[Variational inference](@ref concepts-variational-inference) solves an optimisation problem over a *variational family* ``\mathcal{Q}``. **Constraints specifications** are how you tell `RxInfer` which family to use. They are the main knob by which you trade **accuracy** for **tractability** — and the same knob that unlocks inference in models that would otherwise be intractable.

This page covers the intuition. For the full `@constraints` syntax, macros and options, see the [Constraints Specification manual](@ref user-guide-constraints-specification).

## [Why we need constraints](@id concepts-constraints-specification-why)

Formally, `RxInfer` looks for

```math
q^\ast \;=\; \arg\min_{q(x) \in \mathcal{Q}}\; F[q](\hat{y})\,,
```

where ``F`` is the [Bethe Free Energy](@ref lib-bethe-free-energy). Without any constraints, ``\mathcal{Q}`` is the set of all distributions, and — on a tree-structured [factor graph](@ref concepts-factor-graphs) — the optimum is the exact posterior. Once the graph is loopy, the likelihood is non-conjugate, or the joint posterior simply doesn't admit a closed form, you have two choices:

1. **Accept intractability** and run expensive sampling (e.g. HMC).
2. **Narrow the family** ``\mathcal{Q}`` so that the optimum has a closed form, and optimise inside that narrower family with [message passing](@ref concepts-message-passing).

RxInfer is built around option 2. Constraints specifications are how you narrow ``\mathcal{Q}``.

## [Two flavours of constraints](@id concepts-constraints-specification-flavours)

There are two kinds of assumptions you can impose on ``q``, and `RxInfer` supports both.

### [Factorisation constraints](@id concepts-constraints-specification-factorisation)

Factorisation constraints say *which groups of variables are allowed to carry dependencies in the approximation*. The extremes:

- **Mean-field** — every latent variable is independent:
  ```math
  q(\mu, \tau, x) = q(\mu)\, q(\tau)\, q(x)\,.
  ```
  Cheapest, fastest, but ignores all posterior correlations.
- **Structured** — preserve dependencies between groups of variables that are known (or suspected) to correlate strongly:
  ```math
  q(\mu, \tau, x) = q(\mu, \tau)\, q(x)\,.
  ```
  More accurate, slightly more expensive, and often the right default for the noise-plus-location style models common in signal processing.
- **No factorisation** — pure belief propagation; exact on trees.

In RxInfer these read almost literally like the math:

```julia
using RxInfer

constraints = @constraints begin
    q(μ, τ) = q(μ)q(τ)           # mean-field between μ and τ
end
```

### [Functional form constraints](@id concepts-constraints-specification-form)

Functional form constraints pin down *which distribution family* a marginal is allowed to live in — for example forcing ``q(x)`` to be a `PointMass` (MAP-style estimation), a `Gaussian` (moment matching), or a collection of samples. These are covered in detail in [Built-in Functional Forms](@ref lib-forms).

```julia
using RxInfer

constraints = @constraints begin
    q(x) :: PointMassFormConstraint()   # collapse q(x) to a point estimate
end
```

Together, factorisation and functional-form constraints let you shape the variational family precisely — from fully Bayesian to point-estimate and everything in between — in a single declarative block.

## [A worked intuition](@id concepts-constraints-specification-example)

Consider the canonical IID Normal example — we want to estimate both a mean `μ` and a precision `τ`:

```julia
@model function iid_normal(y)
    μ ~ Normal(mean = 0.0, variance = 1.0)
    τ ~ Gamma(shape = 1.0, rate = 1.0)
    y .~ Normal(mean = μ, precision = τ)
end
```

Here the `Normal` likelihood is *not* conjugate when both `μ` and `τ` are unknown — the exact posterior has no closed form. The pragmatic fix is a mean-field factorisation between `μ` and `τ`:

```julia
constraints = @constraints begin
    q(μ, τ) = q(μ)q(τ)
end
```

This assumption makes each individual update analytic: given `q(τ)`, the update for `q(μ)` is a Gaussian; given `q(μ)`, the update for `q(τ)` is a Gamma. RxInfer alternates between them until the [Bethe Free Energy](@ref lib-bethe-free-energy) converges. You gain tractability at the cost of ignoring the (typically weak) posterior correlation between mean and precision — an excellent trade in practice.

See the [full example](@ref user-guide-constraints-specification-background) for the runnable version, including initialisation and diagnostic output.

## [Guidelines for choosing constraints](@id concepts-constraints-specification-guidelines)

A few rules of thumb when deciding how aggressively to factorise:

- **Start lenient, tighten as needed.** No factorisation on trees; minimal mean-field assumptions on loopy or non-conjugate portions only.
- **Factorise across variable *groups*, not within them.** If two variables are naturally coupled (location–location, scale–scale in the same layer) keep them together.
- **Watch the Bethe Free Energy.** It is your convergence and sanity diagnostic — a BFE that fails to decrease, or that oscillates, often signals over- or under-constrained ``\mathcal{Q}``.
- **Constraints imply initialisation.** A factorised `q` makes inference *iterative*, so you need an initial marginal somewhere in the loop. See [Initialization](@ref initialization) and the examples in the [manual](@ref user-guide-constraints-specification).

## [For deeper understanding](@id concepts-constraints-specification-deeper)

- **[Constraints Specification manual](@ref user-guide-constraints-specification)** — the full `@constraints` macro reference.
- **[Functional Form Constraints](@ref lib-forms)** — `PointMass`, `SampleList`, `FixedMarginal`, custom forms.
- **[Bethe Free Energy](@ref lib-bethe-free-energy)** — the objective constraints reshape.
- **[Variational Message Passing and Local Constraint Manipulation in Factor Graphs](https://doi.org/10.3390/e23070807)** — the theoretical grounding for local constraint manipulation in RxInfer.
