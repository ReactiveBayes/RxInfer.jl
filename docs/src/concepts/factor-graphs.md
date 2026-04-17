# [Factor Graphs](@id concepts-factor-graphs)

A **factor graph** is a picture of how a joint probability distribution factorises into smaller, local pieces. `RxInfer` compiles every `@model` you write into such a graph and performs [Bayesian inference](@ref concepts-bayesian-inference) by exchanging messages along its edges. Understanding what a factor graph is — and how your code maps to one — is the single most useful mental model for using RxInfer productively.

## [Why factorise?](@id concepts-factor-graphs-why)

A generative model defines a joint distribution over all variables. For any non-trivial problem this joint is huge, and manipulating it directly is intractable. Fortunately, real models have *structure*: each local relationship (a prior, a likelihood, a transition, a deterministic equation) usually involves only a handful of variables. Writing the joint as a product of these local pieces,

```math
p(x_1, x_2, \dots, x_n) = \prod_{a} f_a(\mathbf{x}_a)\,,
```

exposes that structure. Every factor ``f_a`` touches only a small subset ``\mathbf{x}_a`` of variables, and that sparsity is what makes [message passing](@ref concepts-message-passing) — and therefore RxInfer — fast.

## [Anatomy of a factor graph](@id concepts-factor-graphs-anatomy)

A factor graph is **bipartite**: it has two kinds of nodes, and edges only run between the two kinds.

- **Variable nodes** ``\bigcirc`` — one per random quantity in the model. They can be *latent* (to be inferred), *observed* (conditioned on data) or *constant* (fixed hyperparameters).
- **Factor nodes** ``\blacksquare`` — one per local function in the factorisation. They can be *stochastic* (a [probability distribution](@ref concepts-probability-distributions) such as a `Normal` or `Gamma`) or *deterministic* (a functional relationship such as `z = x + y`).

An edge connects a factor to every variable it depends on. Messages — themselves probability distributions — flow along these edges during inference.

```
          ┌───────── Beta ─────────┐
          │                         │
         (θ)───────[ Bernoulli ]──(y)
```

The tiny graph above corresponds to `θ ~ Beta(1, 1); y ~ Bernoulli(θ)`: two factor nodes (`Beta` prior and `Bernoulli` likelihood) connected through the shared variable ``\theta``, with the observation ``y`` clamped to its data value.

## [From `@model` code to a factor graph](@id concepts-factor-graphs-spec)

`RxInfer` does not ask you to build this graph by hand. Instead, you describe the model in idiomatic Julia and [`GraphPPL.jl`](https://github.com/ReactiveBayes/GraphPPL.jl) converts it into a factor graph for you. The specification language is intentionally small:

```julia
using RxInfer

@model function state_space_model(y, variance)
    x[1] ~ Normal(mean = 0.0, variance = 100.0)   # prior factor
    y[1] ~ Normal(mean = x[1], variance = variance)
    for i in 2:length(y)
        x[i] ~ Normal(mean = x[i-1], variance = 1.0) # transition factor
        y[i] ~ Normal(mean = x[i],   variance = variance) # likelihood factor
    end
end
```

Each piece of syntax has a precise graph-level meaning:

- `x ~ Distribution(...)` introduces a **stochastic factor** connected to `x` and to the variables appearing in its arguments.
- `z := f(x, y)` (or `z ~ f(x, y)` for random `z`) introduces a **deterministic factor** enforcing `z = f(x, y)`. See [`:=` vs `=`](@ref usage-colon-equality) for why this distinction matters.
- `x .~ Distribution(...)` broadcasts the factor across a collection — convenient for IID data or vectorised observations.
- Regular Julia control flow (`for`, `if`, comprehensions) is evaluated at graph-construction time and unrolls into multiple factor nodes.
- Model arguments can be turned into observations via the `|` conditioning operator, e.g. `model | (y = data,)`.

The full reference — including indexing rules, anonymous nodes, broadcasting and graph visualisation — lives in the [Model Specification](@ref user-guide-model-specification) manual. If you prefer to learn by example, the [Getting started](@ref user-guide-getting-started) guide walks through a complete model from scratch.

!!! tip
    Every `~` statement you write becomes one factor node. Keeping that correspondence in mind makes it much easier to reason about the resulting graph — and to understand why certain [constraints](@ref concepts-constraints-specification) or [initialisations](@ref autoupdates-guide) are required.

## [Trees, loops, and what they imply](@id concepts-factor-graphs-topology)

The *topology* of the graph determines what kind of inference is possible:

- **Tree-structured graphs** (no cycles) admit *exact* inference via belief propagation — a single forward-backward sweep gives you the true marginal posteriors.
- **Graphs with loops** require approximate inference, typically via **loopy belief propagation** or **variational message passing**. RxInfer handles both automatically — see [Message Passing](@ref concepts-message-passing) and [Variational Inference](@ref concepts-variational-inference).

You do not choose the algorithm explicitly: RxInfer inspects the graph and [dispatches](@ref what-is-a-rule) to the appropriate local update rule on every edge.

## [Why this representation pays off](@id concepts-factor-graphs-payoff)

Representing a model as a factor graph buys you three things at once:

1. **Scalability** — local updates mean cost grows with the size of each factor, not with the size of the joint.
2. **Modularity** — adding or swapping a factor is a local change; the rest of the graph keeps working.
3. **Automatic algorithm selection** — conjugate pairs trigger exact rules, non-conjugate regions fall back to variational ones, and deterministic factors compose through the delta trick. You write the model; RxInfer picks the math.

## [For deeper understanding](@id concepts-factor-graphs-deeper)

- **[GraphPPL.jl](https://reactivebayes.github.io/GraphPPL.jl/stable/)** — the model specification front-end and graph construction engine.
- **[ReactiveMP.jl](https://reactivebayes.github.io/ReactiveMP.jl/stable/)** — the low-level factor graph and message passing back-end.
- **[The Factor Graph Approach to Model-Based Signal Processing](https://ieeexplore.ieee.org/document/4282128/)** — Loeliger et al., the canonical introduction to Forney factor graphs.
- **[Factor Graphs and the Sum-Product Algorithm](https://ieeexplore.ieee.org/document/910572)** — Kschischang, Frey and Loeliger (2001).
- **[RxInfer Examples](https://examples.rxinfer.com/)** — gallery of models with visualised factor graphs.
