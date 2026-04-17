# [Bayesian Inference](@id concepts-bayesian-inference)

**Bayesian inference** is the principled way of updating beliefs about unknown quantities when new data arrives. It is the problem `RxInfer` is built to solve — everything else in the documentation (factor graphs, message passing, constraints, reactive execution) is machinery in service of this single goal.

This page gives you the minimum intuition needed to read the rest of the concept pages. For a worked end-to-end example, jump to [Getting started](@ref user-guide-getting-started).

## [The Bayesian recipe](@id concepts-bayesian-inference-recipe)

You start with three ingredients:

1. A **prior** ``p(x)`` — what you believe about the latent (unknown) variables *before* seeing any data.
2. A **likelihood** ``p(y \mid x)`` — how the observed data ``y`` depend on those latent variables.
3. **Observed data** ``\hat{y}``.

Bayes' rule tells you how to combine them into the **posterior** ``p(x \mid \hat{y})`` — your updated belief about ``x`` after seeing the data:

```math
p(x \mid \hat{y}) \;=\; \frac{p(\hat{y} \mid x)\, p(x)}{p(\hat{y})}\,, \qquad
p(\hat{y}) \;=\; \int p(\hat{y} \mid x)\, p(x)\, \mathrm{d}x\,.
```

Read out loud: *"posterior is proportional to likelihood times prior"*. The denominator ``p(\hat{y})`` — the **evidence** or **marginal likelihood** — is just a normalising constant that turns the numerator into a valid [probability distribution](@ref concepts-probability-distributions).

In `RxInfer` you express this recipe once, inside a `@model` block:

```julia
@model function coin(y)
    θ ~ Beta(1.0, 1.0)       # prior over the coin bias
    y .~ Bernoulli(θ)         # likelihood of the flips
end
```

The `infer` function then computes (or approximates) the posterior ``p(\theta \mid \hat{y})`` for you.

## [Why it is hard](@id concepts-bayesian-inference-hard)

The formula above looks innocent, but the evidence integral

```math
p(\hat{y}) = \int p(\hat{y} \mid x)\, p(x)\, \mathrm{d}x
```

is almost always intractable once ``x`` is high-dimensional or the prior and likelihood are not [conjugate](@ref concepts-probability-distributions-conjugate). There are broadly three ways people attack this:

1. **Exact inference** — possible only in special cases (fully conjugate, discrete with small state space, or tree-structured graphs).
2. **Sampling** — Markov-chain Monte Carlo methods such as HMC / NUTS. Flexible but expensive; poor fit for real-time or streaming problems.
3. **Variational inference** — turn the integral into an optimisation problem. This is what RxInfer does, and it is covered on the [Variational Inference](@ref concepts-variational-inference) page.

RxInfer's trick is to combine the speed and scalability of variational inference with *exact* analytical updates wherever conjugacy allows — giving you the best of both worlds. See the [comparison page](@ref comparison) for how this plays out against other tools.

## [Generative models and probabilistic programs](@id concepts-bayesian-inference-generative)

A **generative model** is a joint distribution ``p(x, y)`` describing how both latent variables and observations are produced. Factorising it via the product rule gives

```math
p(x, y) = p(y \mid x)\, p(x)\,,
```

which is exactly the likelihood–prior pairing from above. A **probabilistic program** (like an RxInfer `@model` function) is just executable syntax for writing down ``p(x, y)`` — each `~` statement declares one factor of the joint. This direct correspondence between code and distribution is what makes the [factor graph](@ref concepts-factor-graphs) representation possible.

## [What inference gives you](@id concepts-bayesian-inference-results)

When RxInfer "runs inference" it produces, for each latent variable:

- A **posterior marginal** ``q(x_i) \approx p(x_i \mid \hat{y})`` — typically itself a familiar distribution such as a Gaussian, Gamma or Dirichlet.
- From that marginal you can read off **point estimates** (mean, mode), **uncertainty** (variance, credible intervals), or draw samples for downstream decisions.

Optionally, RxInfer also reports the **[Bethe Free Energy](@ref lib-bethe-free-energy)** — a model-quality score that approximates the (negative log) evidence and is useful for monitoring convergence and comparing models.

## [Where to go next](@id concepts-bayesian-inference-next)

Now that you know *what* Bayesian inference is, the rest of the concept pages explain *how* RxInfer performs it:

- [Factor Graphs](@ref concepts-factor-graphs) — how models are represented internally.
- [Message Passing](@ref concepts-message-passing) — the core inference algorithm operating on the factor graph.
- [Variational Inference](@ref concepts-variational-inference) — the optimisation view that underpins RxInfer.
- [Constraints Specification](@ref concepts-constraints-specification) — how you trade accuracy for tractability.
- [Reactive Programming](@ref concepts-reactive-programming) — the execution model that enables real-time inference.

## [For deeper understanding](@id concepts-bayesian-inference-deeper)

- **Pattern Recognition and Machine Learning** — Christopher Bishop.
- **Bayesian Data Analysis** — Andrew Gelman et al.
- **[An Introduction to Probabilistic Programming](https://arxiv.org/abs/1809.10756)** — van de Meent, Paige, Yang, Wood.
- **[The Factor Graph Approach to Model-Based Signal Processing](https://ieeexplore.ieee.org/document/4282128/)** — Loeliger et al.
