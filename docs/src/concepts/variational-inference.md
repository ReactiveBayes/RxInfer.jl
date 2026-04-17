# [Variational Inference](@id concepts-variational-inference)

**Variational inference (VI)** is the approximation strategy that makes `RxInfer` scale. Exact [Bayesian inference](@ref concepts-bayesian-inference) requires evaluating an integral that is almost always intractable; VI sidesteps the integral entirely by turning inference into an **optimisation problem**.

This page explains the idea at a high level. For the rigorous derivation — including the exact objective RxInfer minimises on a [factor graph](@ref concepts-factor-graphs) — see the [Bethe Free Energy](@ref lib-bethe-free-energy) manual.

## [The core idea](@id concepts-variational-inference-idea)

Instead of computing the true posterior ``p(x \mid \hat{y})`` directly, pick a tractable family of [distributions](@ref concepts-probability-distributions) ``\mathcal{Q}`` — the **variational family** — and search inside it for the member ``q^\ast(x)`` closest to the true posterior:

```math
q^\ast(x) \;=\; \arg\min_{q \in \mathcal{Q}}\; \mathrm{KL}\!\left[q(x)\,\Vert\,p(x \mid \hat{y})\right]\,.
```

The KL divergence measures how "far" one distribution is from another. Minimising it gives you the best in-family approximation of the posterior. Three things are worth noting:

- If ``\mathcal{Q}`` is rich enough to contain the true posterior, the optimum is exact.
- If ``\mathcal{Q}`` is too restrictive, you trade accuracy for tractability — a deliberate, controllable trade-off.
- The KL above depends on the intractable normaliser of the posterior, so it is never minimised directly.

## [Free energy: the tractable objective](@id concepts-variational-inference-free-energy)

A little algebra rewrites the KL divergence into an equivalent objective that *is* computable — the **Variational Free Energy** (VFE):

```math
F[q](\hat{y}) \;=\; \mathbb{E}_{q(x)}\!\left[\log \frac{q(x)}{p(x, \hat{y})}\right]\,.
```

Minimising ``F`` is identical to minimising the KL, up to an additive constant equal to ``-\log p(\hat{y})`` — the log-evidence. So as a side-effect of variational inference, ``-F`` at the optimum gives you an approximation of the log model evidence, which is useful for model comparison and convergence diagnostics.

## [The Bethe approximation](@id concepts-variational-inference-bethe)

Minimising the VFE over arbitrary joint ``q(x)`` is itself intractable on a general model. `RxInfer`'s central trick is to use the **Bethe approximation**, which factorises ``q`` according to the structure of the [factor graph](@ref concepts-factor-graphs) itself:

```math
q(x) \;\triangleq\; \frac{\prod_a q_a(x_a)}{\prod_i q_i(x_i)^{d_i - 1}}\,,
```

where ``q_a`` are factor-local beliefs, ``q_i`` are variable-local beliefs, and ``d_i`` is the degree of variable ``i``. Substituting this into the VFE yields the **Bethe Free Energy** — an objective that *decomposes over the graph* and can therefore be minimised by local [message passing](@ref concepts-message-passing) updates.

This is the deep connection between variational inference and message passing: on a tree, minimising the Bethe Free Energy is exactly belief propagation; on a loopy graph, it is loopy BP; with extra factorisation constraints on ``q``, it is variational message passing. RxInfer implements the unified view.

## [Choosing the variational family](@id concepts-variational-inference-family)

The variational family ``\mathcal{Q}`` is under your control, through [constraints specifications](@ref concepts-constraints-specification). Two common choices:

- **Mean-field** — all latent variables are independent in ``q``: ``q(\mu, \tau) = q(\mu)\, q(\tau)``. Cheapest, but ignores posterior correlations.
- **Structured** — preserves dependencies that matter (e.g. ``q(\mu, \tau) = q(\mu \mid \tau)\, q(\tau)``). More faithful, more compute.
- **No extra constraints** — fall back to plain belief propagation; exact on trees, approximate on loops.

You switch between these with a few lines of `@constraints` syntax — see the [Constraints Specification](@ref concepts-constraints-specification) page for how.

## [What you get back](@id concepts-variational-inference-results)

After the iterations converge, `RxInfer` returns, for each latent variable ``x_i``:

- A **posterior marginal** ``q_i(x_i)`` — in a known distribution family (Gaussian, Gamma, Beta, ...).
- Optionally, the **Bethe Free Energy** trajectory per iteration, which you can monitor to diagnose convergence (see [Convergence and Bethe Free Energy](@ref manual-static-inference-bfe)).

```julia
result = infer(
    model          = my_model(),
    data           = (y = observations,),
    constraints    = my_constraints,
    initialization = my_init,
    iterations     = 20,
    free_energy    = true,
)

posterior_μ = result.posteriors[:μ][end]
bfe         = result.free_energy
```

The whole loop — factorised model, variational family, free energy objective, convergence — is covered end-to-end in [Static Inference](@ref manual-static-inference).

## [For deeper understanding](@id concepts-variational-inference-deeper)

- **[Bethe Free Energy in RxInfer](@ref lib-bethe-free-energy)** — the full derivation of the objective RxInfer minimises.
- **[Constraints Specification](@ref user-guide-constraints-specification)** — how to pick the variational family in practice.
- **[Functional Form Constraints](@ref lib-forms)** — shaping the distribution families of individual marginals.
- **[Variational Message Passing and Local Constraint Manipulation in Factor Graphs](https://doi.org/10.3390/e23070807)** — Şenöz et al.
- **[Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670)** — Blei, Kucukelbir and McAuliffe — an accessible introduction to VI.
- **[The Bethe Free Energy Allows to Compute the Conditional Entropy of Graphical Code Instances](https://arxiv.org/abs/0802.3431)** — Vontobel's classical paper on the Bethe approximation.
