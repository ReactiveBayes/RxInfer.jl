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

You never choose a message update rule by hand. For every message, `RxInfer` finds the rule by:

1. **Node** — which factor you wrote (`Normal`, `Gamma`, `+`, a custom node, ...).
2. **Target** — which edge of the factor the message is heading to.
3. **Algorithm** — the node's algorithm. Most nodes run under the default; the node of a nonlinear function names its approximation.
4. **Inputs** — which messages and marginals arrive on the other edges, and their distribution families. The factorisation your constraints impose decides whether an input is a message, as in BP, or a marginal, as in VMP.

When a [conjugate pair](@ref concepts-probability-distributions-conjugate) meets, the rule is a closed-form update. A Beta prior and a Bernoulli likelihood give an exact Beta posterior:

```@example concepts-message-passing
using RxInfer

@model function coin(y)
    θ ~ Beta(1.0, 1.0)
    for i in eachindex(y)
        y[i] ~ Bernoulli(θ)
    end
end

infer(model = coin(), data = (y = [1.0, 0.0, 1.0, 1.0],)).posteriors[:θ]
```

A nonlinear function of a variable, such as `sin(x)` below, has no closed-form rule. `RxInfer` represents it with a [Delta node](@ref delta-node-manual), and you choose its approximation with `@algorithm`:

```@example concepts-message-passing
@model function sensor(y)
    x ~ Normal(mean = 0.0, variance = 1.0)
    z := sin(x)
    y ~ Normal(mean = z, variance = 0.1)
end

result = infer(
    model = sensor(),
    data = (y = 0.5,),
    algorithm = @algorithm(begin
        sin() -> Linearization()
    end),
)
result.posteriors[:x]
```

When no rule takes the inputs a message needs, as for `sin` without an approximation, inference stops with an error that names the node and the inputs, and says why each candidate rule does not fit. The [Understanding Rules](@ref what-is-a-rule) manual explains how rules are found and what decides their inputs. [Messages by hand](@ref learning-messages-by-hand) and [Variational message passing by hand](@ref learning-vmp-by-hand) call the rules of small models one by one, and [custom rules](@ref create-node) can be added without touching the core engine.

## [Reactive scheduling](@id concepts-message-passing-reactive)

Traditional inference engines compile an explicit *schedule* (forward pass, backward pass, sweep order, ...) before inference begins. `RxInfer` does not. Every node owns a reactive stream of messages, and updates fire whenever their inputs change. The net effect:

- New observations trigger only the messages that actually depend on them.
- The graph is its own scheduler — no global plan to build or maintain.
- Streaming and real-time inference come for free.

The [Reactive Programming](@ref concepts-reactive-programming) concept page expands on this execution model.

## [For deeper understanding](@id concepts-message-passing-deeper)

- **[ReactiveMP.jl](https://reactivebayes.github.io/ReactiveMP.jl/stable/)** — the message passing engine and rule dispatch system.
- **[Understanding Rules](@ref what-is-a-rule)** — how RxInfer picks a rule for every edge.
- **[Messages by hand](@ref learning-messages-by-hand)** and **[Variational message passing by hand](@ref learning-vmp-by-hand)** — belief propagation and variational message passing computed step by step, with the rules RxInfer runs.
- **[Variational Message Passing and Local Constraint Manipulation in Factor Graphs](https://doi.org/10.3390/e23070807)** — Şenöz et al., the theoretical basis of RxInfer's VMP implementation.
- **[Reactive Message Passing for Scalable Bayesian Inference](https://doi.org/10.48550/arXiv.2112.13251)** — scaling message passing with reactive programming.
- **[Factor Graphs and the Sum-Product Algorithm](https://ieeexplore.ieee.org/document/910572)** — Kschischang, Frey and Loeliger (2001).
