# [Message Passing](@id concepts-message-passing)

**Message passing** is the core algorithm that `RxInfer` uses to perform Bayesian inference on factor graphs. Instead of computing full joint distributions (which becomes intractable for large models), RxInfer exchanges local summaries—called **messages**—between connected nodes. Posterior beliefs emerge from combining these messages at each variable node.

## [Belief Propagation (BP)](@id concepts-message-passing-bp)

**Belief propagation**, also known as the sum-product algorithm, computes *exact* marginal posteriors on tree-shaped factor graphs. The fundamental principle: a message from factor `f` toward variable `x` summarizes everything `f` knows about `x` from the rest of the graph:

```math
\mu_{f \to x}(x) = \int f(x, y, z) \; \mu_{y \to f}(y) \; \mu_{z \to f}(z) \; \mathrm{d}y \; \mathrm{d}z
```

The message from `x` back toward `f` collects beliefs arriving at `x` from all *other* connected factors. The marginal posterior `q(x)` is the normalized product of all incoming messages:

```math
q(x) = \frac{1}{Z} \prod_{f \in \text{neighbors}(x)} \mu_{f \to x}(x)
```

On **tree structures** (no cycles), BP converges in a single forward-backward pass with exact results. On graphs with cycles, **loopy belief propagation** iterates until convergence, typically yielding good approximations.

## [Variational Message Passing (VMP)](@id concepts-message-passing-vmp)

**Variational message passing** generalizes BP to perform approximate inference via minimization of the **Bethe Free Energy**—a variational objective rather than exact integrals. RxInfer implements VMP as its primary algorithm because:

1. **Exact BP as special case**: When no factorization constraints exist, VMP reduces to exact belief propagation
2. **Non-conjugate support**: Handles complex models via mean-field or structured factorization assumptions
3. **Local implementation**: Fits naturally with reactive computation model—each message update is self-contained

Under a **mean-field factorization** assumption `q(x, y) = q(x)q(y)`, the VMP message from factor `f` toward variable `x` becomes:

```math
\mu_{f \to x}(x) = \exp \left( \int q(y) \, q(z) \log f(x, y, z) \; \mathrm{d}y \; \mathrm{d}z \right)
```

Key difference from BP: VMP uses *marginals* `q(y)` and `q(z)` rather than messages. RxInfer tracks this distinction through its functional dependencies pipeline, automatically selecting the correct rule based on model constraints.

### Mean-Field vs Structured Factorization

RxInfer supports two factorization assumptions:
- **Mean-field**: Assumes all variables are independent in the approximation (`q(x,y) = q(x)q(y)`)
- **Structured**: Preserves known dependencies between variable groups

The choice affects message computation and convergence properties. RxInfer allows specifying constraints per model to optimize accuracy vs computational cost. Read more about constraints specification [here](@ref user-guide-constraints-specification).

## [Automatic Rule Selection](@id concepts-message-passing-automatic)

RxInfer uses ReactiveMP for message passing, which automatically selects the correct message update rule without user intervention. The dispatch mechanism considers:

1. **Node type**: Stochastic factors use VMP rules; deterministic factors use BP-style updates
2. **Factorization assumption**: Mean-field or structured triggers appropriate VMP variant
3. **Conjugate relationships**: Conjugate likelihood-prior pairs trigger analytical exact updates
4. **Julia multiple dispatch**: Rules are dispatched on node type, outgoing edge, and incoming message types

This automatic selection means adding new factorization assumptions automatically routes computation to correct rules without modifying node code. RxInfer's rule system is extensible—you can add custom rules for your own factor nodes.

### Conjugate vs Non-Conjugate Inference

**Conjugate pairs** trigger exact analytical message updates:
```julia
# Bernoulli likelihood + Beta prior → exact Beta posterior messages
y ~ Bernoulli(θ)
θ ~ Beta(a, b)  # Exact conjugate update computed automatically
```

**Non-conjugate pairs** use variational approximation:
```julia
# Poisson likelihood + Gaussian prior → non-conjugate, VMP approximation
y ~ Poisson(λ)
λ ~ Normal(μ, σ²)  # Requires numerical integration in VMP rule
```

RxInfer's accuracy advantage comes from exploiting conjugate relationships when available—delivering exact posteriors without hyperparameter tuning.

## [The Reactive Message Passing Model](@id concepts-message-passing-reactive)

The word *reactive* refers to how messages are **scheduled and propagated**. Unlike traditional inference engines that build explicit computation schedules (forward-backward passes) before inference starts, RxInfer takes a different approach: **no pre-built schedule**.

Instead:
- Each variable and factor node holds a **reactive stream** (`MessageObservable` or `MarginalObservable`) emitting updated values when inputs change
- When new data arrives via observation, changes propagate automatically through the graph
- Propagation order determined by **graph structure at runtime**, not static plan
- Only rules depending on updated values are triggered—minimal computation

This reactive design enables:
- **Real-time inference**: New observations trigger incremental updates without full recomputation
- **Online learning**: Streaming data continuously updates posteriors
- **Efficiency**: Only affected nodes recompute, saving computational resources

Read more about reactive message passing and its implementation [here](https://reactivebayes.github.io/ReactiveMP.jl/stable/).

## [For Deeper Understanding](@id concepts-message-passing-deeper)

To explore message passing in more depth:

- **[ReactiveMP.jl Message Passing](https://reactivebayes.github.io/ReactiveMP.jl/stable/)** — Low-level engine perspective on VMP algorithm and reactive computation model
- **[Rocket.jl Documentation](https://github.com/ReactiveBayes/Rocket.jl)** — Reactive programming with observables, streams, and triggers underlying RxInfer's message propagation
- **[Variational Message Passing and Local Constraint Manipulation in Factor Graphs](https://doi.org/10.3390/e23070807)** — Theoretical aspects of VMP and constraint handling
- **[Reactive Message Passing for Scalable Bayesian Inference](https://doi.org/10.48550/arXiv.2112.13251)** — Implementation details and benchmark comparisons

For understanding how RxInfer achieves superior accuracy over HMC methods, see the [PhD dissertation](https://pure.tue.nl/ws/portalfiles/portal/313860204/20231219_Bagaev_hf.pdf) outlining core principles behind reactive probabilistic programming.
