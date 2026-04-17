# [Message Passing](@id concepts-message-passing)

**Message passing** is the algorithm `RxInfer` uses to turn a [factor graph](@ref concepts-factor-graphs) into concrete posterior distributions. Instead of ever touching the full joint, nodes exchange small local summaries — **messages** — along edges, and the posterior over each variable emerges from combining them.

The intuition is simple: *every message from a factor towards a variable is a local answer to the question "given everything I know about the rest of the graph, what do I think this variable looks like?"*.

## [Belief Propagation (BP)](@id concepts-message-passing-bp)

The classical message passing algorithm is **belief propagation**, also known as the **sum-product algorithm**. A message from factor ``f`` toward variable ``x`` integrates ``f`` against all other incoming messages:

```math
\mu_{f \to x}(x) \;=\; \int f(x, y, z)\, \mu_{y \to f}(y)\, \mu_{z \to f}(z)\, \mathrm{d}y\, \mathrm{d}z\,.
```

The message back from ``x`` collects beliefs from all *other* factors attached to ``x``. The marginal posterior is then the normalised product of incoming messages at a variable node:

```math
q(x) \;=\; \frac{1}{Z} \prod_{f \in \text{nb}(x)} \mu_{f \to x}(x)\,.
```

On **tree-shaped graphs** this procedure is *exact*: a single forward-backward sweep produces the true posterior marginals. On graphs with cycles, iterating the same updates — **loopy BP** — yields a principled approximation that is often excellent in practice.

## [Variational Message Passing (VMP)](@id concepts-message-passing-vmp)

Once your model has loops, non-conjugate factors, or you deliberately impose simplifying assumptions, exact BP is no longer available. `RxInfer`'s primary algorithm is **variational message passing**, which turns inference into minimisation of the [Bethe Free Energy](@ref lib-bethe-free-energy) — the variational objective described in full on the [Variational Inference](@ref concepts-variational-inference) page.

Under a mean-field-style factorisation, the factor-to-variable message becomes

```math
\mu_{f \to x}(x) \;=\; \exp\!\left( \int q(y)\, q(z)\, \log f(x, y, z)\, \mathrm{d}y\, \mathrm{d}z \right)\,,
```

where ``q(y)`` and ``q(z)`` are the *current marginal beliefs* about the neighbouring variables. Two things make VMP attractive:

1. **BP is a special case**: with no extra factorisation constraints, VMP reduces to ordinary belief propagation.
2. **Locality**: each update still depends only on immediate neighbours, so the reactive execution model scales naturally.

Which factorisation you get — full mean-field, structured, or none at all — is controlled by [constraints specifications](@ref concepts-constraints-specification).

## [Automatic rule selection](@id concepts-message-passing-automatic)

You never choose a message update rule by hand. `RxInfer` uses Julia's multiple dispatch to pick the right rule for every edge based on:

1. **Node type** — which factor you wrote (`Normal`, `Gamma`, a custom deterministic node, ...).
2. **Outgoing edge** — which argument of the factor the message is heading towards.
3. **Incoming message types** — the distribution families arriving on the other edges.
4. **Factorisation assumption** — whether the surrounding constraints require a BP-style or VMP-style rule.

When a [conjugate pair](@ref concepts-probability-distributions-conjugate) is detected, the dispatched rule is a closed-form analytical update. Non-conjugate combinations fall back to numerical or approximate rules. The [Understanding Rules](@ref what-is-a-rule) manual explains exactly how this machinery works under the hood, and [custom rules](@ref create-node) can be added without touching the core engine.

```julia
# Conjugate — dispatches to an exact analytical Beta update
θ ~ Beta(1.0, 1.0)
y ~ Bernoulli(θ)

# Non-conjugate — dispatches to a variational approximation
λ ~ Normal(mean = 0.0, variance = 1.0)
y ~ Poisson(exp(λ))
```

## [Reactive scheduling](@id concepts-message-passing-reactive)

Traditional inference engines compile an explicit *schedule* (forward pass, backward pass, sweep order, ...) before inference begins. `RxInfer` does not. Every node owns a reactive stream of messages, and updates fire whenever their inputs change. The net effect:

- New observations trigger only the messages that actually depend on them.
- The graph is its own scheduler — no global plan to build or maintain.
- Streaming and real-time inference come for free.

The [Reactive Programming](@ref concepts-reactive-programming) concept page expands on this execution model.

## [For deeper understanding](@id concepts-message-passing-deeper)

- **[ReactiveMP.jl](https://reactivebayes.github.io/ReactiveMP.jl/stable/)** — the message passing engine and rule dispatch system.
- **[Understanding Rules](@ref what-is-a-rule)** — how RxInfer picks a rule for every edge.
- **[Variational Message Passing and Local Constraint Manipulation in Factor Graphs](https://doi.org/10.3390/e23070807)** — Şenöz et al., the theoretical basis of RxInfer's VMP implementation.
- **[Reactive Message Passing for Scalable Bayesian Inference](https://doi.org/10.48550/arXiv.2112.13251)** — scaling message passing with reactive programming.
- **[Factor Graphs and the Sum-Product Algorithm](https://ieeexplore.ieee.org/document/910572)** — Kschischang, Frey and Loeliger (2001).
