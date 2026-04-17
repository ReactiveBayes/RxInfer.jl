# [Probability Distributions](@id concepts-probability-distributions)

Probability distributions are the vocabulary of Bayesian inference. Every variable in an `RxInfer` model — observations, latent states, parameters, priors — is described by a distribution, and the job of [Bayesian inference](@ref concepts-bayesian-inference) is to reshape these distributions in light of data.

This page is a short refresher. If you have already worked with probabilistic programming, feel free to skim it and move on to [Bayesian Inference](@ref concepts-bayesian-inference) and [Factor Graphs](@ref concepts-factor-graphs).

## [Random variables and distributions](@id concepts-probability-distributions-basics)

A **random variable** ``x`` is a quantity whose value is uncertain. A **probability distribution** ``p(x)`` assigns plausibilities to the values ``x`` can take. For continuous variables we work with densities; for discrete variables with probability mass functions. Either way the total probability integrates (or sums) to one:

```math
\int p(x)\, \mathrm{d}x = 1\,.
```

A **joint distribution** ``p(x, y)`` describes two or more variables together. The **conditional distribution** ``p(x \mid y)`` describes ``x`` once ``y`` is known. These two objects are tied together by the product rule:

```math
p(x, y) = p(x \mid y)\, p(y) = p(y \mid x)\, p(x)\,.
```

That simple identity is the seed of [Bayes' rule](@ref concepts-bayesian-inference).

## [Families you will see most often](@id concepts-probability-distributions-families)

In practice you will keep running into a handful of distributions. RxInfer supports all standard distributions from [`Distributions.jl`](https://juliastats.org/Distributions.jl/stable/) and many additional specialized forms. The most common ones are:

- **Gaussian / Normal** ``\mathcal{N}(\mu, \sigma^2)`` — the default for continuous real-valued variables; closed under linear operations, which is why it dominates state-space models and signal processing.
- **Gamma / Inverse-Gamma** — standard priors on positive quantities such as precisions and variances.
- **Beta** — priors on probabilities and rates, bounded on ``[0, 1]``.
- **Bernoulli / Categorical** — single binary or discrete outcomes.
- **Dirichlet** — priors on categorical probability vectors.
- **Wishart / Inverse-Wishart** — priors on covariance and precision matrices.
- **Poisson** — counts of events per unit time.

Each family comes with its own parameters (mean, variance, shape, rate, ...), and RxInfer lets you use whichever parameterisation is natural for your problem (see [Distributions.jl compatibility](@ref user-guide-model-specification-distributions)).

## [Conjugate pairs](@id concepts-probability-distributions-conjugate)

Two distributions form a **conjugate pair** when a likelihood from one family, multiplied by a prior from a matching family, yields a posterior in the *same* family as the prior. In that case the posterior update has a closed-form analytical expression — no numerical integration, no sampling, no gradient descent.

Classic examples:

| Likelihood  | Conjugate prior      | Resulting posterior  |
|-------------|----------------------|----------------------|
| Bernoulli   | Beta                 | Beta                 |
| Categorical | Dirichlet            | Dirichlet            |
| Gaussian (known variance) | Gaussian | Gaussian          |
| Gaussian (known mean)     | Gamma    | Gamma             |
| Poisson     | Gamma                | Gamma                |

RxInfer leans heavily on conjugacy: whenever a [factor graph](@ref concepts-factor-graphs) contains conjugate pairs, [message passing](@ref concepts-message-passing) dispatches to exact analytical update rules automatically. This is one of the main reasons inference in RxInfer is fast *and* accurate. See the [Wikipedia overview of conjugate priors](https://en.wikipedia.org/wiki/Conjugate_prior) for more examples.

## [The exponential family](@id concepts-probability-distributions-exponential-family)

Most of the distributions above belong to the **exponential family**, which has the general form

```math
p(x \mid \theta) = h(x)\, \exp\!\left( \eta(\theta)^\top T(x) - A(\theta) \right)\,,
```

with sufficient statistics ``T(x)``, natural parameters ``\eta(\theta)`` and log-partition ``A(\theta)``. Exponential-family distributions are convenient because:

- messages stay within the family under conjugate updates,
- natural parameters add when you multiply two densities,
- many variational update rules reduce to simple updates of natural parameters.

You rarely need to write things in this form yourself, but it is the machinery that makes RxInfer's [variational message passing](@ref concepts-variational-inference) rules compact and efficient.

## [Where distributions live in RxInfer](@id concepts-probability-distributions-rxinfer)

Inside a model specification, distributions appear on the right-hand side of the `~` operator:

```julia
μ ~ Normal(mean = 0.0, variance = 100.0)
τ ~ Gamma(shape = 1.0, rate = 1.0)
y ~ Normal(mean = μ, precision = τ)
```

Each `~` statement introduces a factor node in the underlying [factor graph](@ref concepts-factor-graphs). During inference, messages flowing over the edges of that graph *are themselves distributions* from these same families — Gaussians, Gammas, Betas, and so on.

## [For deeper understanding](@id concepts-probability-distributions-deeper)

- **[Distributions.jl](https://juliastats.org/Distributions.jl/stable/)** — the Julia ecosystem's reference catalogue of probability distributions.
- **[ExponentialFamily.jl](https://github.com/ReactiveBayes/ExponentialFamily.jl)** — the exponential-family backbone used by RxInfer for efficient message representations.
- **[Conjugate prior (Wikipedia)](https://en.wikipedia.org/wiki/Conjugate_prior)** — table of conjugate likelihood/prior pairs.
- **Pattern Recognition and Machine Learning** by Christopher Bishop — classic textbook treatment of distributions, exponential families and conjugacy.
