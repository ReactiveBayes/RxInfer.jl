# [Factor Graphs](@id concepts-factor-graphs)

A **factor graph** is a bipartite graphical representation of how a joint probability distribution factorizes into a product of local functions. In `RxInfer`, factor graphs serve as the foundational structure for automatic Bayesian inference, enabling efficient message-passing computations across complex probabilistic models.

## [What is a Factor Graph?](@id concepts-factor-graphs-what)

Factor graphs provide a visual and computational representation of probabilistic relationships between variables. They decompose a joint distribution into local factors, each representing a conditional probability or deterministic constraint:

```math
p(x_1, x_2, \dots, x_n) = \prod_{i} f_i(\text{variables}_i)
```

Each **factor** `f_i` only depends on a subset of variables, creating sparse connections that enable efficient local computations. This factorization is key to RxInfer's scalability—instead of computing the full joint distribution (which becomes intractable for large models), we perform localized message updates across the graph.

## [Nodes and Edges](@id concepts-factor-graphs-nodes)

Factor graphs contain two distinct types of nodes connected by edges:

### Variable Nodes `(○)`

Variable nodes represent random quantities in your model:
- **Latent variables** — unknown quantities to be inferred (e.g., hidden states, parameters)
- **Observed variables** — measured data fixed during inference
- **Constant variables** — known hyperparameters or fixed values

### Factor Nodes `[□]`

Factor nodes represent local functions connecting variables:
- **Stochastic factors** — probability distributions (likelihoods, priors)
- **Deterministic factors** — hard constraints (e.g., `z = x + y`, `σ² = μ`)

Stochastic factors add probabilistic relationships and contribute to the Bethe Free Energy objective. Deterministic factors enforce exact functional relationships without adding uncertainty.

## [GraphPPL and Factor Graph Construction](@id concepts-factor-graphs-graphppl)

`RxInfer` leverages [`GraphPPL.jl`](https://github.com/ReactiveBayes/GraphPPL.jl) for intuitive model specification. The `@model` macro converts probabilistic syntax into factor graphs. Read more about model specification [here](@ref user-guide-model-specification).

## [Conjugate Pairs and Analytical Posteriors](@id concepts-factor-graphs-conjugate)

RxInfer exploits **conjugate likelihood-prior pairs** for exact analytical posteriors, significantly improving accuracy and speed. When a prior and likelihood are conjugate, the posterior belongs to the same distribution family. Read more about conjugate pairs [here](https://en.wikipedia.org/wiki/Conjugate_prior).

## [For Deeper Understanding](@id concepts-factor-graphs-deeper)

To explore factor graphs in more depth:

- **[ReactiveMP.jl Factor Graphs]**(https://reactivebayes.github.io/ReactiveMP.jl/) — Low-level engine perspective on variable and factor node types
- **[GraphPPL.jl Documentation]**(https://reactivebayes.github.io/GraphPPL.jl/stable/) — Model specification syntax and visualization tools
- **[The Factor Graph Approach to Model-Based Signal Processing]**(https://ieeexplore.ieee.org/document/4282128/) — Introduction to message passing and Forney factor graphs
- **[RxInfer Examples]**(https://examples.rxinfer.com/)
