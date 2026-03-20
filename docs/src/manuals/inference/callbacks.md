# [Callbacks](@id manual-inference-callbacks)

```@meta
CurrentModule = RxInfer
```

The [`infer`](@ref) function and the underlying reactive message passing engine both have their own lifecycle, consisting of multiple steps.
By supplying callbacks, users can inject custom logic at specific moments during the inference procedure — for example, for [debugging](@ref user-guide-debugging-callbacks), [performance analysis](@ref manual-inference-benchmark-callbacks), or [early stopping](@ref manual-inference-early-stopping).

## Event-based callback system

All callbacks in RxInfer use an **event-based dispatch** system built on `ReactiveMP.Event{E}`. Each callback event is a concrete struct that carries all relevant data as named fields. This makes callbacks self-documenting and extensible.

For example, an `AfterIterationEvent` has fields `model` and `iteration`:

```julia
# NamedTuple callback — receives the event struct
callbacks = (
    after_iteration = (event) -> println("Iteration ", event.iteration, " done"),
)
```

## Callback types

The `callbacks` keyword argument of the [`infer`](@ref) function accepts three types of callback handlers:

### `NamedTuple`

The simplest way to define callbacks is via a `NamedTuple`, where keys correspond to event names and values are functions that receive a single **event object**:

```@example manual-inference-callbacks
using RxInfer
using RxInfer.ReactiveMP
using Test #hide

@model function coin_model(y)
    θ ~ Beta(1, 1)
    y .~ Bernoulli(θ)
end

before_inference_called = Ref(false) #hide
after_inference_called = Ref(false) #hide

result = infer(
    model = coin_model(),
    data  = (y = [1, 0, 1, 1, 0],),
    callbacks = (
        before_inference   = (event) -> begin
            before_inference_called[] = true #hide
            println("Starting inference on model: ", typeof(event.model))
        end,
        after_inference    = (event) -> begin
            after_inference_called[] = true #hide
            println("Inference completed")
        end,
    )
)

@test before_inference_called[] #hide
@test after_inference_called[] #hide
nothing #hide
```

!!! warning
    When defining a `NamedTuple` with a single entry, make sure to include a trailing comma.
    In Julia, `(key = value)` is parsed as a variable assignment, **not** a `NamedTuple`.
    Use `(key = value,)` (with trailing comma) instead. `ReactiveMP` will raise a helpful error if this mistake is detected.

### `Dict`

A `Dict{Symbol}` works the same way as a `NamedTuple`, but allows dynamic construction of callbacks:

```@example manual-inference-callbacks
my_callbacks = Dict(
    :before_inference => (event) -> println("Starting inference"),
    :after_inference  => (event) -> println("Inference completed"),
)

result = infer(
    model = coin_model(),
    data  = (y = [1, 0, 1, 1, 0],),
    callbacks = my_callbacks,
)

@test result.posteriors[:θ] isa Distribution #hide
nothing #hide
```

### Custom callback structures

For more advanced use cases, you can pass any custom structure as a callback handler.
The structure must implement `ReactiveMP.invoke_callback` methods for the event types it wants to handle:

```@example manual-inference-callbacks
struct MyCallbackHandler
    log::Vector{String}
end

# Catch-all: ignore events you don't care about
ReactiveMP.invoke_callback(::MyCallbackHandler, ::ReactiveMP.Event) = nothing

# Handle specific events by dispatching on the concrete event type
function ReactiveMP.invoke_callback(handler::MyCallbackHandler, event::BeforeInferenceEvent)
    push!(handler.log, "inference started")
end

function ReactiveMP.invoke_callback(handler::MyCallbackHandler, event::AfterInferenceEvent)
    push!(handler.log, "inference completed")
end

handler = MyCallbackHandler(String[])

result = infer(
    model = coin_model(),
    data  = (y = [1, 0, 1, 1, 0],),
    callbacks = handler,
)

@test length(handler.log) == 2 #hide
@test handler.log[1] == "inference started" #hide
@test handler.log[2] == "inference completed" #hide
println(handler.log)
```

#### Dispatching by event name with `Event{:name}`

Since every event struct is a subtype of `ReactiveMP.Event{E}` where `E` is a `Symbol`, you can also dispatch on `ReactiveMP.Event{:event_name}` instead of the concrete type name. This is equivalent — you still have access to all the same fields — and can be more convenient when the concrete type is not exported. For example, ReactiveMP-level events like `BeforeMessageRuleCallEvent` are not exported by default, but you can always dispatch on `ReactiveMP.Event{:before_message_rule_call}`:

```@example manual-inference-callbacks
struct MyEventNameHandler
    log::Vector{String}
end

# Catch-all
ReactiveMP.invoke_callback(::MyEventNameHandler, ::ReactiveMP.Event) = nothing

# Dispatch using Event{:name} — no need to know the concrete struct name
function ReactiveMP.invoke_callback(handler::MyEventNameHandler, event::ReactiveMP.Event{:before_inference})
    push!(handler.log, "inference started (via Event name)")
end

function ReactiveMP.invoke_callback(handler::MyEventNameHandler, event::ReactiveMP.Event{:after_inference})
    push!(handler.log, "inference completed (via Event name)")
end

handler_by_name = MyEventNameHandler(String[])

result = infer(
    model = coin_model(),
    data  = (y = [1, 0, 1, 1, 0],),
    callbacks = handler_by_name,
)

@test length(handler_by_name.log) == 2 #hide
@test handler_by_name.log[1] == "inference started (via Event name)" #hide
@test handler_by_name.log[2] == "inference completed (via Event name)" #hide
println(handler_by_name.log)
```

Both approaches are fully interchangeable — use whichever is more convenient for your use case.

Custom callback structures are useful when you need to:
- Maintain state across events (e.g., collecting timing information)
- Implement complex logic that spans multiple events
- Store information in the model's `metadata` dictionary for later access

RxInfer provides built-in callback structures such as [`RxInferBenchmarkCallbacks`](@ref) and [`StopEarlyIterationStrategy`](@ref) as examples of this pattern.

## Model metadata

The [`ProbabilisticModel`](@ref) structure contains a `metadata` dictionary (`Dict{Any, Any}`) that callbacks can use to store arbitrary information during inference. This is accessible from the inference result via `result.model.metadata`.

For example, you can track the history of marginal updates during variational inference:

```@example manual-inference-callbacks
@model function gaussian_model(y)
    μ ~ Normal(mean = 0.0, variance = 100.0)
    τ ~ Gamma(shape = 1.0, rate = 1.0)
    y .~ Normal(mean = μ, precision = τ)
end

struct MarginalHistoryCollector end

ReactiveMP.invoke_callback(::MarginalHistoryCollector, ::ReactiveMP.Event) = nothing

function ReactiveMP.invoke_callback(::MarginalHistoryCollector, event::OnMarginalUpdateEvent)
    history = get!(() -> [], event.model.metadata, :marginal_history)
    push!(history, (iteration_variable = event.variable_name, value = event.update))
end

result = infer(
    model = gaussian_model(),
    data  = (y = randn(50),),
    constraints = MeanField(),
    initialization = @initialization(begin
        q(μ) = vague(NormalMeanVariance)
        q(τ) = vague(GammaShapeRate)
    end),
    iterations = 5,
    callbacks = MarginalHistoryCollector(),
)

@test haskey(result.model.metadata, :marginal_history) #hide
@test length(result.model.metadata[:marginal_history]) > 0 #hide

# Access the collected marginal history
history = result.model.metadata[:marginal_history]
println("Collected ", length(history), " marginal updates across all iterations")
println("Variables updated: ", unique(map(h -> h.iteration_variable, history)))
```


## Available events

Callbacks can listen to events from two layers: **RxInfer-level** events from the inference lifecycle, and **ReactiveMP-level** events from the message passing engine itself.

```@eval
using RxInfer, Test
# Verify that the events documented below match the actual available callbacks
@test RxInfer.available_callbacks(RxInfer.batch_inference) === Val((
    :on_marginal_update,
    :before_model_creation,
    :after_model_creation,
    :before_inference,
    :before_iteration,
    :before_data_update,
    :after_data_update,
    :after_iteration,
    :after_inference,
    :before_message_rule_call,
    :after_message_rule_call,
    :before_product_of_two_messages,
    :after_product_of_two_messages,
    :before_product_of_messages,
    :after_product_of_messages,
    :before_form_constraint_applied,
    :after_form_constraint_applied,
    :before_marginal_computation,
    :after_marginal_computation
))
@test RxInfer.available_callbacks(RxInfer.streaming_inference) === Val((
    :before_model_creation,
    :after_model_creation,
    :before_autostart,
    :after_autostart,
    :before_message_rule_call,
    :after_message_rule_call,
    :before_product_of_two_messages,
    :after_product_of_two_messages,
    :before_product_of_messages,
    :after_product_of_messages,
    :before_form_constraint_applied,
    :after_form_constraint_applied,
    :before_marginal_computation,
    :after_marginal_computation
))
nothing
```

### RxInfer events

These events are fired by the [`infer`](@ref) function during the inference lifecycle.
Each event is a concrete struct subtyping `ReactiveMP.Event{E}` with named fields.

#### Common to batch and streamline inference

[`BeforeModelCreationEvent`](@ref) — `:before_model_creation`

Called before the model is created. Has no fields.

[`AfterModelCreationEvent`](@ref) — `:after_model_creation`

Called after the model has been created.
- `model`: the created [`ProbabilisticModel`](@ref) instance

#### Batch inference only

For more details on batch inference, see [Static inference](@ref manual-static-inference).

[`OnMarginalUpdateEvent`](@ref) — `:on_marginal_update`

Called after each marginal update.
- `model`: the [`ProbabilisticModel`](@ref) instance
- `variable_name`: the name of the updated variable (`Symbol`)
- `update`: the updated marginal value

[`BeforeInferenceEvent`](@ref) — `:before_inference`

Called before the inference procedure starts.
- `model`: the [`ProbabilisticModel`](@ref) instance

[`AfterInferenceEvent`](@ref) — `:after_inference`

Called after the inference procedure ends.
- `model`: the [`ProbabilisticModel`](@ref) instance

[`BeforeIterationEvent`](@ref) — `:before_iteration`

Called before each variational iteration.
- `model`: the [`ProbabilisticModel`](@ref) instance
- `iteration`: the current iteration number (`Int`)

[`AfterIterationEvent`](@ref) — `:after_iteration`

Called after each variational iteration.
- `model`: the [`ProbabilisticModel`](@ref) instance
- `iteration`: the current iteration number (`Int`)

!!! note
    `before_iteration` and `after_iteration` callbacks can return [`StopIteration()`](@ref) to halt iterations early. Any other return value (including `nothing`) will let iterations continue. See [Early stopping](@ref manual-inference-early-stopping) for an example.

[`BeforeDataUpdateEvent`](@ref) — `:before_data_update`

Called before each data update.
- `model`: the [`ProbabilisticModel`](@ref) instance
- `data`: the data being used

[`AfterDataUpdateEvent`](@ref) — `:after_data_update`

Called after each data update.
- `model`: the [`ProbabilisticModel`](@ref) instance
- `data`: the data that was used

#### Streamline inference only

For more details on streamline inference, see [Streamline inference](@ref manual-online-inference).

[`BeforeAutostartEvent`](@ref) — `:before_autostart`

Called before `RxInfer.start()`, if `autostart` is set to `true`.
- `engine`: the [`RxInferenceEngine`](@ref) instance

[`AfterAutostartEvent`](@ref) — `:after_autostart`

Called after `RxInfer.start()`, if `autostart` is set to `true`.
- `engine`: the [`RxInferenceEngine`](@ref) instance

### ReactiveMP events

These lower-level events are fired by the `ReactiveMP` message passing engine during inference. They are available in both batch and streamline inference modes. Each event is a concrete struct subtyping `ReactiveMP.Event{E}` with named fields — refer to the ReactiveMP documentation for field details.

- `BeforeMessageRuleCallEvent` / `AfterMessageRuleCallEvent` — fired around message rule computations
- `BeforeProductOfTwoMessagesEvent` / `AfterProductOfTwoMessagesEvent` — fired around pairwise message products
- `BeforeProductOfMessagesEvent` / `AfterProductOfMessagesEvent` — fired around folded message products
- `BeforeFormConstraintAppliedEvent` / `AfterFormConstraintAppliedEvent` — fired around form constraint application
- `BeforeMarginalComputationEvent` / `AfterMarginalComputationEvent` — fired around marginal computations

For detailed descriptions of these events and their fields, refer to the official documentation of `ReactiveMP`.

## Migration from positional-argument callbacks

!!! warning "Breaking change"
    Previous versions of RxInfer passed callback arguments positionally:
    ```julia
    # OLD (no longer works)
    callbacks = (
        after_iteration = (model, iteration) -> println(iteration),
        on_marginal_update = (model, name, update) -> println(name),
    )
    ```
    The new system uses **event structs** with named fields. Each callback now receives a single event object:
    ```julia
    # NEW
    callbacks = (
        after_iteration = (event) -> println(event.iteration),
        on_marginal_update = (event) -> println(event.variable_name),
    )
    ```

    For custom callback structures, implement `invoke_callback` methods that dispatch on the concrete event type:
    ```julia
    # Dispatch on specific events
    ReactiveMP.invoke_callback(::MyHandler, event::BeforeInferenceEvent) = ... # event.model
    # Catch-all for events you don't care about
    ReactiveMP.invoke_callback(::MyHandler, ::ReactiveMP.Event) = nothing
    ```

    The migration is straightforward: replace positional arguments with named field access on the event object. The event structs are fully documented with their field names — see the sections above.

## Built-in callback handlers

RxInfer provides the following built-in callback structures:

- [`RxInferBenchmarkCallbacks`](@ref) — collects timing statistics across inference runs. See [Benchmark callbacks](@ref manual-inference-benchmark-callbacks).
- [`RxInferTraceCallbacks`](@ref) — records all callback events for debugging and inspection. See [Trace callbacks](@ref manual-inference-trace-callbacks).
- [`StopEarlyIterationStrategy`](@ref) — stops variational iterations early based on free energy convergence. See [Early stopping](@ref manual-inference-early-stopping).

## Event type reference

```@docs
BeforeModelCreationEvent
AfterModelCreationEvent
BeforeInferenceEvent
AfterInferenceEvent
BeforeIterationEvent
AfterIterationEvent
BeforeDataUpdateEvent
AfterDataUpdateEvent
OnMarginalUpdateEvent
BeforeAutostartEvent
AfterAutostartEvent
```
