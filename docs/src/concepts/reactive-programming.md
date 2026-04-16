# [Reactive Programming Model](@id concepts-reactive-programming)

`RxInfer` is built on a **reactive programming** paradigm. Unlike traditional inference engines that follow pre-defined, static computation schedules (e.g., performing forward and backward passes), RxInfer operates by **reacting to changes** in underlying data. This paradigm enables real-time Bayesian inference with minimal computational overhead.

## [The Mental Model: Streams vs Values](@id concepts-reactive-programming-mental)

To use RxInfer effectively, shift your thinking from **"static values"** to **"dynamic streams"**. This mental model is fundamental to understanding how messages propagate through the factor graph.

### 1. Observables as Information Streams

In traditional algorithms, a message is just a piece of data at a specific point in time. In `RxInfer`, messages and marginals are treated as **Observables**—dynamic streams of information:

```
Think of an Observable not as:
❌ A single number (e.g., mean = 3.14)

Think of it as:
✅ A stream of values over time [3.14, 3.15, 3.12, 3.16, ...]
```

Whenever a node performs computation and produces a new result, it **"emits"** this value into the stream. Any downstream node listening to this stream automatically receives the update.

### 2. Dependency-Driven Execution

Because nodes are connected via streams, the graph handles its own execution. You don't manually trigger "message passing steps":

```
External event occurs (new observation added)
    ↓
Change triggers specific node update
    ↓
Node's output changes → notifies all connected neighbors
    ↓
Change propagates through graph structure
    ↓
Only nodes actually affected by the update recompute
```

This **dependency-driven execution** ensures minimum computation necessary to keep beliefs up-to-date. Nodes disconnected from the changed variable remain untouched—no wasted computation.

## [Real-Time Inference Benefits](@id concepts-reactive-programming-benefits)

The reactive paradigm enables capabilities impractical with traditional engines:

### Incremental Updates

When new observations arrive, RxInfer performs **incremental updates** rather than full recomputation.
This makes RxInfer ideal for:
- **Streaming data** (sensor readings, real-time signals)
- **Online learning** (continuously updating models)
- **Real-time applications** (autonomous vehicles, audio processing)

### Schedule-Free Execution

Traditional engines build explicit schedules before inference:
```
Build schedule → Execute forward pass → Execute backward pass → Repeat
```

RxInfer has **no pre-built schedule**. Propagation order determined by graph structure at runtime. This flexibility enables:
- Adaptive computation based on actual changes
- No overhead for scheduling unused nodes
- Natural support for dynamic model structures

## [Rocket.jl Integration](@id concepts-reactive-programming-rocket)

The reactive machinery underlying RxInfer comes from [`Rocket.jl`](https://github.com/ReactiveBayes/Rocket.jl)—a Julia library for reactive programming with observables. Rocket provides:
- **Observables** — streams that emit values over time
- **Subscribers** — nodes listening to observable changes
- **Operators** — transformations on streams (map, filter, reduce)

RxInfer builds on these primitives: each factor and variable node holds `MessageObservable` or `MarginalObservable` streams from ReactiveMP (which is also built on top of Rocket). When new data arrives, Rocket's reactive engine automatically propagates changes through subscribed nodes.

## [For Deeper Understanding](@id concepts-reactive-programming-deeper)

To explore reactive programming in more depth:

- **[Rocket.jl Documentation](https://github.com/ReactiveBayes/Rocket.jl)** — Low-level mechanics of observables, and reactive streams in Julia. Essential for understanding RxInfer's reactive machinery
- **[ReactiveMP.jl Reactive Message Passing](https://reactivebayes.github.io/ReactiveMP.jl/stable/)** — Engine perspective on how messages are scheduled via reactive streams
- **[RxInfer Examples: Real-Time Inference](https://examples.rxinfer.com)** — Practical applications of reactive message passing in streaming scenarios

For understanding the broader paradigm, see [Reactive Probabilistic Programming for Scalable Bayesian Inference](https://pure.tue.nl/ws/portalfiles/portal/313860204/20231219_Bagaev_hf.pdf)—the PhD dissertation outlining core ideas behind RxInfer's reactive approach.
