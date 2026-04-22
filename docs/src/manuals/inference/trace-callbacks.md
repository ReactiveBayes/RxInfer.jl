# [Trace callbacks](@id manual-inference-trace-callbacks)

```@meta
CurrentModule = RxInfer
```

`RxInfer` provides a built-in callback structure called [`RxInferTraceCallbacks`](@ref) for recording all callback events during the inference procedure.
Each event is stored as a [`TracedEvent`](@ref) containing the event name (as a `Symbol`) and the event object itself.
This is useful for debugging, understanding the inference flow, and inspecting what happens at each stage.
For general information about the callbacks system, see [Callbacks](@ref manual-inference-callbacks).

## Basic usage

```@example manual-inference-trace-callbacks
using RxInfer
using Test #hide

@model function iid_normal(y)
    μ  ~ Normal(mean = 0.0, variance = 100.0)
    γ  ~ Gamma(shape = 1.0, rate = 1.0)
    y .~ Normal(mean = μ, precision = γ)
end

init = @initialization begin
    q(μ) = vague(NormalMeanVariance)
end

# Create a trace callbacks instance
trace = RxInferTraceCallbacks()

result = infer(
    model = iid_normal(),
    data = (y = randn(10),),
    constraints = MeanField(),
    iterations = 3,
    initialization = init,
    callbacks = trace,
)

events = RxInfer.tracedevents(trace)
@test !isempty(events) #hide
@test all(e -> e isa TracedEvent, events) #hide

println("Recorded $(length(events)) events")
for i in 1:10
    println("  ", events[i])
end
println("...")
```

## Using `trace = true`

Instead of creating a `RxInferTraceCallbacks` instance manually, you can use the `trace = true` keyword argument in the [`infer`](@ref) function.
This automatically merges a `RxInferTraceCallbacks` instance with any user-provided callbacks and saves it to the model's metadata:

```@example manual-inference-trace-callbacks
result = infer(
    model = iid_normal(),
    data = (y = randn(10),),
    constraints = MeanField(),
    iterations = 3,
    initialization = init,
    trace = true,
)

@test haskey(result.model.metadata, :trace) #hide
@test result.model.metadata[:trace] isa RxInferTraceCallbacks #hide

trace = result.model.metadata[:trace]
events = RxInfer.tracedevents(trace)
println("Recorded $(length(events)) events via trace = true")
```

## Accessing from model metadata

After model creation, the trace callbacks instance is automatically saved into the model's metadata under the `:trace` key.
This makes it accessible from the inference result without needing to hold onto the callbacks object separately:

```@example manual-inference-trace-callbacks
result = infer(
    model = iid_normal(),
    data = (y = randn(10),),
    constraints = MeanField(),
    iterations = 3,
    initialization = init,
    callbacks = RxInferTraceCallbacks(),
)

@test haskey(result.model.metadata, :trace) #hide
@test result.model.metadata[:trace] isa RxInferTraceCallbacks #hide

trace = result.model.metadata[:trace]
events = RxInfer.tracedevents(trace)
println("Recorded $(length(events)) events via trace = true")
```

## Inspecting traced events

Each [`TracedEvent`](@ref) has a single field:
- `event::ReactiveMP.Event` — the original event object that was passed to the callback

You can retrieve the event name via `ReactiveMP.event_name(typeof(traced_event.event))` and access event-specific fields directly on `traced_event.event`.

```@example manual-inference-trace-callbacks
using RxInfer.ReactiveMP: event_name
events = RxInfer.tracedevents(trace)

# Filter for specific events
iteration_events = filter(e -> event_name(typeof(e.event)) === :before_iteration, events)
@test length(iteration_events) == 3 #hide
println("Number of iterations: ", length(iteration_events))
```

## Combining with other callbacks

`trace = true` is compatible with other callbacks, including `benchmark = true` and custom callbacks:

```@example manual-inference-trace-callbacks
result = infer(
    model = iid_normal(),
    data = (y = randn(10),),
    constraints = MeanField(),
    iterations = 3,
    initialization = init,
    trace = true,
    benchmark = true,
)

@test haskey(result.model.metadata, :trace) #hide
@test haskey(result.model.metadata, :benchmark) #hide

println("Trace included: ", haskey(result.model.metadata, :trace))
println("Benchmark included: ", haskey(result.model.metadata, :benchmark))
```

## API Reference

```@docs
RxInferTraceCallbacks
TracedEvent
RxInfer.tracedevents
```
