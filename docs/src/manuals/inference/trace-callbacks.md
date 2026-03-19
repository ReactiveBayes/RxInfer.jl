# [Trace callbacks](@id manual-inference-trace-callbacks)

```@meta
CurrentModule = RxInfer
```

`RxInfer` provides a built-in callback structure called [`RxInferTraceCallbacks`](@ref) for recording all callback events during the inference procedure.
Each event is stored as a [`TracedEvent`](@ref) containing the event name (as a `Symbol`) and the arguments passed to the callback.
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
for event in events
    println("  ", event)
end
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
println(trace)
```

## Inspecting traced events

Each [`TracedEvent`](@ref) has two fields:
- `event::Symbol` — the name of the callback event (e.g. `:before_iteration`, `:after_model_creation`)
- `arguments::Tuple` — the arguments that were passed to the callback

```@example manual-inference-trace-callbacks
events = RxInfer.tracedevents(trace)

# Filter for specific events
iteration_events = filter(e -> e.event === :before_iteration, events)
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

println("Trace: ", result.model.metadata[:trace])
println("Benchmark: ", result.model.metadata[:benchmark])
```

## API Reference

```@docs
RxInferTraceCallbacks
TracedEvent
RxInfer.tracedevents
```
