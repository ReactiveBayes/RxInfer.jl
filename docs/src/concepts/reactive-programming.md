# [Reactive Programming](@id concepts-reactive-programming)

`RxInfer` is built around a **reactive programming** paradigm: inference is not a batch procedure you invoke, but a *graph of dependencies* that continuously reacts to new information. This is what gives RxInfer its characteristic strengths — real-time updates, streaming data support, and schedule-free execution — and what most clearly sets it apart from sampling- or autodiff-based inference engines.

If you have used reactive frameworks such as RxJS, SwiftUI or Rocket.jl, the mental model will feel familiar. If not, the key shift is summarised on this page.

## [Streams, not values](@id concepts-reactive-programming-streams)

In a classical inference engine, a "message" or "marginal" is a *value* you compute at a specific moment. In `RxInfer`, it is a **stream of values over time** — an observable.

```
Classical view               ┆  Reactive view
────────────────────────────  ┆  ─────────────────────────────────────────
message = compute(...)       ┆  message_stream : [m₀, m₁, m₂, m₃, ...]
(one snapshot, recompute)    ┆  (keeps emitting as inputs change)
```

Every variable and factor node in the [factor graph](@ref concepts-factor-graphs) owns such a stream. When a stream emits a new value, all downstream nodes subscribed to it react — re-running the local [message passing](@ref concepts-message-passing) update and, in turn, emitting into their own streams. Posteriors update automatically as long as new observations keep arriving.

## [Dependency-driven execution](@id concepts-reactive-programming-execution)

Because the graph is a web of subscriptions, *you do not schedule inference*. The data flow decides what gets recomputed and when:

1. A new observation is pushed into the graph.
2. Only the nodes whose inputs actually changed fire their update rules.
3. Changes propagate outward, hop by hop, through subscribed neighbours.
4. Nodes unaffected by the update stay idle — zero wasted work.

The outcome is two-fold. You get **incremental updates** (no full recomputation when a single observation arrives) and **schedule-free execution** (no forward/backward pass to plan in advance). Both are essential for the real-time and streaming workloads RxInfer targets — see [Streaming (online) inference](@ref manual-online-inference) for the streaming inference API.

## [What reactivity enables](@id concepts-reactive-programming-benefits)

Practical capabilities that come almost for free out of the reactive model:

- **Real-time inference** — sensor readings, signal processing, robotics control: observations flow in, posteriors flow out.
- **Online learning** — long-running models that keep refining themselves as data arrives, without periodic re-fitting.
- **Selective updates** — only the part of the graph touched by new data recomputes, which matters a lot on large state-space models.
- **Natural support for auto-updates** — posteriors from one time step can be wired as priors for the next through [autoupdates](@ref autoupdates-guide).

## [Rocket.jl under the hood](@id concepts-reactive-programming-rocket)

The reactive machinery comes from [`Rocket.jl`](https://github.com/ReactiveBayes/Rocket.jl), Julia's library for observables and streams. RxInfer and [ReactiveMP](https://github.com/ReactiveBayes/ReactiveMP.jl) wrap Rocket's primitives — observables, subscribers, operators — into the domain-specific streams used by inference:

- `MessageObservable` — a stream of messages flowing along one direction of an edge.
- `MarginalObservable` — a stream of posterior marginals at a variable node.

You rarely touch these directly, but they are what make the whole system tick. Browsing the Rocket documentation is the fastest way to build a deep intuition for how RxInfer actually executes.

## [For deeper understanding](@id concepts-reactive-programming-deeper)

- **[Rocket.jl](https://github.com/ReactiveBayes/Rocket.jl)** — the reactive-programming foundation.
- **[ReactiveMP.jl](https://reactivebayes.github.io/ReactiveMP.jl/stable/)** — how messages are scheduled via reactive streams.
- **[Reactive Message Passing for Scalable Bayesian Inference](https://doi.org/10.48550/arXiv.2112.13251)** — scalability and real-time properties of the approach.
- **[Reactive Probabilistic Programming for Scalable Bayesian Inference](https://pure.tue.nl/ws/portalfiles/portal/313860204/20231219_Bagaev_hf.pdf)** — PhD dissertation outlining the core ideas behind RxInfer's reactive approach.
- **[Streaming (online) inference](@ref manual-online-inference)** — user-facing API for streaming and real-time scenarios.
