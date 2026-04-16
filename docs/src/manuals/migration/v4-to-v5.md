# Migration Guide from version 4.x to 5.x

This guide explains how to migrate your code from `RxInfer` version 4.x to 5.x. The main breaking changes in version 5.x are:

1. **Addons** have been renamed to **annotations** (propagated from the new `ReactiveMP` v6 release).
2. The **callback system** has been refactored to use event structs instead of dispatch on positional arguments.
3. The default postprocessing strategy has been simplified.
4. **Pipeline stages** and the per-node **scheduler** argument have been replaced by a unified **stream postprocessor** API (propagated from `ReactiveMP` v6).

## Addons have been renamed to annotations

In `ReactiveMP` v6, the "addon" system was redesigned and renamed to "annotations". RxInfer has been updated to match this new API. All user-facing references to addons have been renamed to annotations.

### The `addons` keyword in `infer` is now `annotations`

The most visible change is the keyword argument in the [`infer`](@ref) function:

**Before (v4.x):**
```julia
result = infer(
    model = my_model(),
    data  = (y = observations,),
    addons = AddonLogScale(),
)
```

**After (v5.x):**
```julia
result = infer(
    model = my_model(),
    data  = (y = observations,),
    annotations = LogScaleAnnotations(),
)
```

The same rename applies when passing the option through the `options` NamedTuple:

**Before (v4.x):**
```julia
infer(model = my_model(), data = ..., options = (addons = AddonLogScale(),))
```

**After (v5.x):**
```julia
infer(model = my_model(), data = ..., options = (annotations = LogScaleAnnotations(),))
```

### Addon types have been renamed

The two built-in addon types from `ReactiveMP` have been renamed:

| Old name (v4.x) | New name (v5.x)              |
| --------------- | ---------------------------- |
| `AddonLogScale` | `LogScaleAnnotations`        |
| `AddonMemory`   | `InputArgumentsAnnotations`  |

For example, to trace input arguments to message rules (previously the "Memory addon"), use:

**Before (v4.x):**
```julia
result = infer(
    model = coin_model(),
    data  = (x = dataset,),
    addons = (AddonMemory(),),
)
RxInfer.ReactiveMP.getaddons(result.posteriors[:θ])
```

**After (v5.x):**
```julia
result = infer(
    model = coin_model(),
    data  = (x = dataset,),
    annotations = (InputArgumentsAnnotations(),),
)
RxInfer.ReactiveMP.getannotations(result.posteriors[:θ])
```

### `getaddons` / `setaddons` have been renamed

If you used `RxInfer.getaddons` or `RxInfer.setaddons` on `ReactiveMPInferenceOptions` directly, rename them to `getannotations` / `setannotations`. The same applies to `ReactiveMP.getaddons` / `ReactiveMP.AbstractAddon`, which are now `ReactiveMP.getannotations` / `ReactiveMP.AbstractAnnotations`.

### The `Marginal` type and constructor have changed

In `ReactiveMP` v6, the `Marginal` type no longer carries addons as a type parameter. The new signature is `Marginal{D}` instead of `Marginal{D, A}`, and annotations are always stored in a `ReactiveMP.AnnotationDict` (which is empty when no annotations are enabled).

**Constructor changes:**

- **No annotations**: use the 3-argument form `Marginal(data, is_point, is_clamped)` instead of the previous `Marginal(data, is_point, is_clamped, nothing)`.
- **With annotations**: use `Marginal(data, is_point, is_clamped, annotation_dict)` where `annotation_dict::ReactiveMP.AnnotationDict`.

The `AnnotationDict` itself only supports an empty constructor. To populate it, use `ReactiveMP.annotate!`:

```julia
ann = ReactiveMP.AnnotationDict()
ReactiveMP.annotate!(ann, :my_key, my_value)
m   = Marginal(data, false, false, ann)
```

This is mostly relevant for advanced users who construct `Marginal` instances directly (e.g. in custom postprocessing tests). Most users will not encounter this change.

## Callbacks now receive event structs

The callback system has been completely refactored. In v4.x, callbacks were dispatched with positional arguments (e.g. `(model, iteration) -> ...`). In v5.x, every callback receives a single **event object** — a concrete struct subtyping `ReactiveMP.Event{E}` with named fields.

This change unifies how RxInfer-level and ReactiveMP-level callbacks are handled, makes the API extensible, and allows passing custom callback handlers as structs.

### NamedTuple/Dict callbacks

If you used a `NamedTuple` or `Dict` to register callbacks by name, simply replace positional arguments with named field access on the event object.

**Before (v4.x):**
```julia
result = infer(
    model = my_model(),
    data  = (y = observations,),
    callbacks = (
        before_iteration = (model, iteration) -> println("Starting iteration ", iteration),
        after_iteration  = (model, iteration) -> println("Finished iteration ", iteration),
    ),
)
```

**After (v5.x):**
```julia
result = infer(
    model = my_model(),
    data  = (y = observations,),
    callbacks = (
        before_iteration = (event) -> println("Starting iteration ", event.iteration),
        after_iteration  = (event) -> println("Finished iteration ", event.iteration),
    ),
)
```

### New event types

The following RxInfer-level event types are available in v5.x:

- `BeforeModelCreationEvent`, `AfterModelCreationEvent`
- `BeforeInferenceEvent`, `AfterInferenceEvent`
- `BeforeIterationEvent`, `AfterIterationEvent`
- `BeforeDataUpdateEvent`, `AfterDataUpdateEvent`
- `OnMarginalUpdateEvent`
- `BeforeAutostartEvent`, `AfterAutostartEvent`

In addition, RxInfer now exposes ReactiveMP-level callbacks such as `before_message_rule_call`, `after_message_rule_call`, `before_product_of_messages`, `after_product_of_messages`, `before_marginal_computation`, `after_marginal_computation`, and others. See the [Callbacks](@ref manual-inference-callbacks) section for the full list and event field definitions.

### Custom callback handlers

The `callbacks` field of [`infer`](@ref) now also accepts any custom struct that implements `ReactiveMP.handle_event`. This is useful if you want to share state across callbacks or implement complex logic in a single, type-stable handler:

```julia
struct MyCallbackHandler
    log::Vector{String}
end

function ReactiveMP.handle_event(handler::MyCallbackHandler, event::AfterIterationEvent)
    push!(handler.log, "iteration $(event.iteration) finished")
end

handler = MyCallbackHandler(String[])
result  = infer(
    model = my_model(),
    data  = (y = observations,),
    callbacks = handler,
)
```

### Early stopping with `stop_iteration`

In v4.x, the `before_iteration` / `after_iteration` callbacks could halt iterations by returning `true`. In v5.x, this is now done by setting a mutable `stop_iteration::Bool` field on the event object (default `false`).

**Before (v4.x):**
```julia
callbacks = (
    after_iteration = (model, iteration) -> begin
        return iteration >= 5  # stop after iteration 5
    end,
)
```

**After (v5.x):**
```julia
callbacks = (
    after_iteration = (event) -> begin
        if event.iteration >= 5
            event.stop_iteration = true
        end
    end,
)
```

The built-in [`StopEarlyIterationStrategy`](@ref) has been updated accordingly — it now receives an `AfterIterationEvent` instead of `(model, iteration)`. If you used `StopEarlyIterationStrategy` directly, no changes are required.

See [Early stopping](@ref manual-inference-early-stopping) for a complete example.

## `DefaultPostprocess` has been removed

In v4.x, the `postprocess` keyword in [`infer`](@ref) defaulted to `DefaultPostprocess()`, which inspected each result at runtime to decide whether to unpack the `Marginal` wrapper. In v5.x this logic has been simplified:

- The `postprocess` keyword now defaults to `nothing`.
- When `postprocess === nothing`, [`infer`](@ref) automatically picks a strategy based on the `annotations` keyword:
  - If `annotations === nothing` (the default), [`UnpackMarginalPostprocess`](@ref) is used — the `Marginal` wrapper is stripped from results.
  - If annotations are enabled, [`NoopPostprocess`](@ref) is used — the wrapper is preserved so that annotation data remains accessible via `ReactiveMP.getannotations`.
- If you pass `postprocess = ...` explicitly, your choice is always respected.

### Migration

If you were relying on the default behavior, no changes are needed — the new automatic selection produces the same results in both the no-annotations and with-annotations cases.

If you explicitly passed `DefaultPostprocess()`, simply remove it (or replace with `nothing`):

**Before (v4.x):**
```julia
result = infer(
    model = my_model(),
    data  = (y = observations,),
    postprocess = DefaultPostprocess(),
)
```

**After (v5.x):**
```julia
result = infer(
    model = my_model(),
    data  = (y = observations,),
    # postprocess defaults to `nothing`, which selects the right strategy automatically
)
```

Custom postprocessing strategies (any user-defined type that implements [`inference_postprocess`](@ref)) continue to work unchanged.

See [Inference results postprocessing](@ref user-guide-inference-postprocess) for more details on the postprocessing system.

## Pipeline stages and the node-level scheduler have been replaced by stream postprocessors

In `ReactiveMP` v6, the `AbstractPipelineStage` hierarchy and the per-node `scheduler` argument have been unified into a single `ReactiveMP.AbstractStreamPostprocessor` abstraction that postprocesses outbound message streams, marginal streams, and score streams uniformly. `RxInfer` exposes this through the `stream_postprocessors` option instead of the old `pipeline` / `scheduler` options.

### The `where { pipeline = ... }` node clause has been removed

Attaching pipeline stages per node via the `where { pipeline = ... }` clause is no longer supported — the API it depended on (`AbstractPipelineStage`, `LoggerPipelineStage`, `AsyncPipelineStage`, `DiscontinuePipelineStage`, `ScheduleOnPipelineStage`, `EmptyPipelineStage`, `CompositePipelineStage`, `apply_pipeline_stage`, `schedule_updates`, ...) was removed in `ReactiveMP` v6.

**Before (v4.x):**
```julia
@model function my_model(y)
    μ ~ Normal(mean = 0.0, variance = 100.0)
    γ ~ Gamma(shape = 1.0, rate = 1.0)
    y .~ Normal(mean = μ, precision = γ) where { pipeline = LoggerPipelineStage() }
end
```

**After (v5.x):**

`LoggerPipelineStage` has no direct replacement — use the new [Callbacks](@ref manual-inference-callbacks) system (e.g. `before_message_rule_call` / `after_message_rule_call`) to observe every message computation without subscribing to the reactive streams. See [Tracing individual message computations](@ref user-guide-debugging-message-computations) and [Trace callbacks](@ref manual-inference-trace-callbacks).

### The `options = (scheduler = ...,)` keyword is now `stream_postprocessors`

The `scheduler` option under `infer(..., options = ...)` has been renamed to `stream_postprocessors` and now expects a `ReactiveMP.AbstractStreamPostprocessor` (or `nothing`) rather than a Rocket.jl scheduler. To reproduce the old `schedule_on(scheduler)` behaviour, wrap the scheduler in a `ReactiveMP.ScheduleOnStreamPostprocessor`:

**Before (v4.x):**
```julia
infer(
    model = my_model(),
    data  = (y = dataset,),
    options = (scheduler = PendingScheduler(),),
)
```

**After (v5.x):**
```julia
infer(
    model = my_model(),
    data  = (y = dataset,),
    options = (stream_postprocessors = ReactiveMP.ScheduleOnStreamPostprocessor(PendingScheduler()),),
)
```

The default is now `nothing` (no-op pass-through) instead of `AsapScheduler()`. This matches the `RandomVariableActivationOptions` / `FactorNodeActivationOptions` default in `ReactiveMP` v6.

### The `limit_stack_depth` option is unchanged

`options = (limit_stack_depth = N,)` still works exactly as before. Internally it is now expanded to `ReactiveMP.ScheduleOnStreamPostprocessor(RxInfer.LimitStackScheduler(N))` instead of being passed through as a scheduler directly. The behaviour is identical; you only need to change your code if you combined `limit_stack_depth` with an explicit `scheduler` (replace the latter with `stream_postprocessors`).

### Pipeline stage replacements

| v4.x                                   | v5.x                                                                                                 |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `LoggerPipelineStage`                  | [Callbacks](@ref manual-inference-callbacks) (`before_message_rule_call` / `after_message_rule_call`) |
| `AsyncPipelineStage`                   | `ReactiveMP.ScheduleOnStreamPostprocessor(AsyncScheduler())`                                         |
| `ScheduleOnPipelineStage(sched)`       | `ReactiveMP.ScheduleOnStreamPostprocessor(sched)`                                                    |
| `DiscontinuePipelineStage`             | Removed (was unused); implement a custom `AbstractStreamPostprocessor` if needed                      |
| `EmptyPipelineStage()`                 | `nothing`                                                                                            |
| `CompositePipelineStage(stages)`       | `ReactiveMP.CompositeStreamPostprocessor(stages)`                                                    |
| `schedule_updates(vars; pipeline_stage = ...)` | Pass a `ReactiveMP.ScheduleOnStreamPostprocessor` through the activation options instead       |

See the `ReactiveMP` v5→v6 migration guide and the `Stream postprocessors` page in the `ReactiveMP` documentation for the full low-level API.

## Summary of breaking changes

**Annotations:**
- `addons` keyword argument → `annotations` (in `infer`, `batch_inference`, `streaming_inference`, and `options` NamedTuple)
- `AddonLogScale` → `LogScaleAnnotations`
- `AddonMemory` → `InputArgumentsAnnotations`
- `getaddons` / `setaddons` → `getannotations` / `setannotations`
- `ReactiveMP.AbstractAddon` → `ReactiveMP.AbstractAnnotations`
- `Marginal{D, A}` → `Marginal{D}` (annotations stored in an `AnnotationDict`)

**Callbacks:**
- Callbacks receive a single event object instead of positional arguments
- New event types: `BeforeModelCreationEvent`, `AfterModelCreationEvent`, `BeforeInferenceEvent`, `AfterInferenceEvent`, `BeforeIterationEvent`, `AfterIterationEvent`, `BeforeDataUpdateEvent`, `AfterDataUpdateEvent`, `OnMarginalUpdateEvent`, `BeforeAutostartEvent`, `AfterAutostartEvent`
- Early stopping uses `event.stop_iteration = true` instead of returning `true`
- `callbacks` field also accepts custom structs implementing `ReactiveMP.handle_event`

**Postprocessing:**
- `DefaultPostprocess` removed; `postprocess` defaults to `nothing` and is auto-selected based on `annotations`

**Pipeline stages and scheduler:**
- `where { pipeline = ... }` node clause removed; pipeline-stage types (`LoggerPipelineStage`, `AsyncPipelineStage`, `ScheduleOnPipelineStage`, `DiscontinuePipelineStage`, `EmptyPipelineStage`, `CompositePipelineStage`, `AbstractPipelineStage`) no longer exist
- `options = (scheduler = ...,)` → `options = (stream_postprocessors = ReactiveMP.ScheduleOnStreamPostprocessor(...),)`; default changed from `AsapScheduler()` to `nothing`
- `LoggerPipelineStage` has no direct replacement — use `before_message_rule_call` / `after_message_rule_call` callbacks instead
- `limit_stack_depth` continues to work unchanged

For a deeper dive into the new annotation processor system and how to implement custom annotations, see the `ReactiveMP.jl` documentation.
