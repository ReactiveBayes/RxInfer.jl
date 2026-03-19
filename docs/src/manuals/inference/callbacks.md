# [Callbacks](@id manual-inference-callbacks)

```@meta
CurrentModule = RxInfer
```

The [`infer`](@ref) function and the underlying reactive message passing engine both have their own lifecycle, consisting of multiple steps.
By supplying callbacks, users can inject custom logic at specific moments during the inference procedure — for example, for [debugging](@ref user-guide-debugging-callbacks), [performance analysis](@ref manual-inference-benchmark-callbacks), or [early stopping](@ref manual-inference-early-stopping).

## Callback types

The `callbacks` keyword argument of the [`infer`](@ref) function accepts three types of callback handlers:

### `NamedTuple`

The simplest way to define callbacks is via a `NamedTuple`, where keys correspond to event names and values are functions:

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
        before_inference   = (model) -> begin
            before_inference_called[] = true #hide
            println("Starting inference")
        end,
        after_inference    = (model) -> begin
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
    :before_inference => (model) -> println("Starting inference"),
    :after_inference  => (model) -> println("Inference completed"),
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
The structure must implement `ReactiveMP.invoke_callback` methods for the events it wants to handle:

```@example manual-inference-callbacks
struct MyCallbackHandler
    log::Vector{String}
end

# Catch-all: ignore events you don't care about
ReactiveMP.invoke_callback(::MyCallbackHandler, event, args...) = nothing

# Handle specific events
function ReactiveMP.invoke_callback(handler::MyCallbackHandler, ::Val{:before_inference}, model)
    push!(handler.log, "inference started")
end

function ReactiveMP.invoke_callback(handler::MyCallbackHandler, ::Val{:after_inference}, model)
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

ReactiveMP.invoke_callback(::MarginalHistoryCollector, event, args...) = nothing

function ReactiveMP.invoke_callback(::MarginalHistoryCollector, ::Val{:on_marginal_update}, model, name, update)
    history = get!(() -> [], model.metadata, :marginal_history)
    push!(history, (iteration_variable = name, value = update))
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

#### Common to batch and streamline inference

```julia
before_model_creation()
```
Called before the model is created. Does not receive any arguments.

```julia
after_model_creation(model::ProbabilisticModel)
```
Called after the model has been created. Receives the `model`.

#### Batch inference only

For more details on batch inference, see [Static inference](@ref manual-static-inference).

```julia
on_marginal_update(model::ProbabilisticModel, name::Symbol, update)
```
Called after each marginal update. Receives the `model`, the variable `name`, and the updated marginal.

```julia
before_inference(model::ProbabilisticModel)
```
Called before the inference procedure starts.

```julia
after_inference(model::ProbabilisticModel)
```
Called after the inference procedure ends.

```julia
before_iteration(model::ProbabilisticModel, iteration::Int)
```
Called before each variational iteration.

```julia
after_iteration(model::ProbabilisticModel, iteration::Int)
```
Called after each variational iteration.

!!! note
    `before_iteration` and `after_iteration` callbacks can return [`StopIteration()`](@ref) to halt iterations early. Any other return value (including `nothing`) will let iterations continue. See [Early stopping](@ref manual-inference-early-stopping) for an example.

```julia
before_data_update(model::ProbabilisticModel, data)
```
Called before each data update.

```julia
after_data_update(model::ProbabilisticModel, data)
```
Called after each data update.

#### Streamline inference only

For more details on streamline inference, see [Streamline inference](@ref manual-online-inference).

```julia
before_autostart(engine::RxInferenceEngine)
```
Called before `RxInfer.start()`, if `autostart` is set to `true`.

```julia
after_autostart(engine::RxInferenceEngine)
```
Called after `RxInfer.start()`, if `autostart` is set to `true`.

### ReactiveMP events

These lower-level events are fired by the `ReactiveMP` message passing engine during inference. They are available in both batch and streamline inference modes.

- `before_message_rule_call` / `after_message_rule_call` — fired around message rule computations
- `before_product_of_two_messages` / `after_product_of_two_messages` — fired around pairwise message products
- `before_product_of_messages` / `after_product_of_messages` — fired around folded message products
- `before_form_constraint_applied` / `after_form_constraint_applied` — fired around form constraint application
- `before_marginal_computation` / `after_marginal_computation` — fired around marginal computations

For detailed descriptions of these events and their arguments, refer to the official documentation of `ReactiveMP`.

## Built-in callback handlers

RxInfer provides the following built-in callback structures:

- [`RxInferBenchmarkCallbacks`](@ref) — collects timing statistics across inference runs. See [Benchmark callbacks](@ref manual-inference-benchmark-callbacks).
- [`StopEarlyIterationStrategy`](@ref) — stops variational iterations early based on free energy convergence. See [Early stopping](@ref manual-inference-early-stopping).
