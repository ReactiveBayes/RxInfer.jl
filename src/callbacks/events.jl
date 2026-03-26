export BeforeModelCreationEvent, AfterModelCreationEvent
export BeforeInferenceEvent, AfterInferenceEvent
export BeforeIterationEvent, AfterIterationEvent
export BeforeDataUpdateEvent, AfterDataUpdateEvent
export OnMarginalUpdateEvent
export BeforeAutostartEvent, AfterAutostartEvent

import ReactiveMP: Event, event_name
using UUIDs

## RxInfer-level callback event types
## These events subtype `ReactiveMP.Event{E}` and carry relevant data as fields.

"""
    BeforeModelCreationEvent <: ReactiveMP.Event{:before_model_creation}

Fires right before the probabilistic model is created in the [`infer`](@ref) function.

# Fields
- `trace_id`: a `UUID` shared with the corresponding [`AfterModelCreationEvent`](@ref)

See also: [`AfterModelCreationEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct BeforeModelCreationEvent <: Event{:before_model_creation}
    trace_id::UUID
end

"""
    AfterModelCreationEvent{M} <: ReactiveMP.Event{:after_model_creation}

Fires right after the probabilistic model is created in the [`infer`](@ref) function.

# Fields
- `model::M`: the created [`ProbabilisticModel`](@ref) instance
- `trace_id`: a `UUID` shared with the corresponding [`BeforeModelCreationEvent`](@ref)

See also: [`BeforeModelCreationEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct AfterModelCreationEvent{M} <: Event{:after_model_creation}
    model::M
    trace_id::UUID
end

"""
    BeforeInferenceEvent{M} <: ReactiveMP.Event{:before_inference}

Fires right before the inference procedure starts (after model creation and subscription setup).

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `trace_id`: a `UUID` shared with the corresponding [`AfterInferenceEvent`](@ref)

See also: [`AfterInferenceEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct BeforeInferenceEvent{M} <: Event{:before_inference}
    model::M
    trace_id::UUID
end

"""
    AfterInferenceEvent{M} <: ReactiveMP.Event{:after_inference}

Fires right after the inference procedure completes.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `trace_id`: a `UUID` shared with the corresponding [`BeforeInferenceEvent`](@ref)

See also: [`BeforeInferenceEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct AfterInferenceEvent{M} <: Event{:after_inference}
    model::M
    trace_id::UUID
end

"""
    BeforeIterationEvent{M} <: ReactiveMP.Event{:before_iteration}

Fires right before each variational iteration.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `iteration::Int`: the current iteration number
- `stop_iteration::Bool`: set to `true` from a callback to halt iterations early (default: `false`)
- `trace_id`: a `UUID` shared with the corresponding [`AfterIterationEvent`](@ref)

See also: [`AfterIterationEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
mutable struct BeforeIterationEvent{M} <: Event{:before_iteration}
    model::M
    iteration::Int
    stop_iteration::Bool
    trace_id::UUID
end

BeforeIterationEvent(model::M, iteration::Int) where {M} = BeforeIterationEvent(model, iteration, false, uuid4())

"""
    AfterIterationEvent{M} <: ReactiveMP.Event{:after_iteration}

Fires right after each variational iteration.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `iteration::Int`: the current iteration number
- `stop_iteration::Bool`: set to `true` from a callback to halt iterations early (default: `false`)
- `trace_id`: a `UUID` shared with the corresponding [`BeforeIterationEvent`](@ref)

See also: [`BeforeIterationEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
mutable struct AfterIterationEvent{M} <: Event{:after_iteration}
    model::M
    iteration::Int
    stop_iteration::Bool
    trace_id::UUID
end

AfterIterationEvent(model::M, iteration::Int, trace_id::UUID) where {M} = AfterIterationEvent(model, iteration, false, trace_id)

"""
    BeforeDataUpdateEvent{M, D} <: ReactiveMP.Event{:before_data_update}

Fires right before updating data variables in each iteration.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `data::D`: the data being used for the update
- `trace_id`: a `UUID` shared with the corresponding [`AfterDataUpdateEvent`](@ref)

See also: [`AfterDataUpdateEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct BeforeDataUpdateEvent{M, D} <: Event{:before_data_update}
    model::M
    data::D
    trace_id::UUID
end

"""
    AfterDataUpdateEvent{M, D} <: ReactiveMP.Event{:after_data_update}

Fires right after updating data variables in each iteration.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `data::D`: the data that was used for the update
- `trace_id`: a `UUID` shared with the corresponding [`BeforeDataUpdateEvent`](@ref)

See also: [`BeforeDataUpdateEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct AfterDataUpdateEvent{M, D} <: Event{:after_data_update}
    model::M
    data::D
    trace_id::UUID
end

"""
    OnMarginalUpdateEvent{M, U} <: ReactiveMP.Event{:on_marginal_update}

Fires each time a marginal posterior for a variable is updated during inference.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `variable_name::Symbol`: the name of the variable whose marginal was updated
- `update::U`: the updated marginal value

See also: [Callbacks](@ref manual-inference-callbacks)
"""
struct OnMarginalUpdateEvent{M, U} <: Event{:on_marginal_update}
    model::M
    variable_name::Symbol
    update::U
end

"""
    BeforeAutostartEvent{E} <: ReactiveMP.Event{:before_autostart}

Fires right before `RxInfer.start()` is called on the streaming inference engine (when `autostart = true`).

# Fields
- `engine::E`: the [`RxInferenceEngine`](@ref) instance
- `trace_id`: a `UUID` shared with the corresponding [`AfterAutostartEvent`](@ref)

See also: [`AfterAutostartEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct BeforeAutostartEvent{E} <: Event{:before_autostart}
    engine::E
    trace_id::UUID
end

"""
    AfterAutostartEvent{E} <: ReactiveMP.Event{:after_autostart}

Fires right after `RxInfer.start()` is called on the streaming inference engine (when `autostart = true`).

# Fields
- `engine::E`: the [`RxInferenceEngine`](@ref) instance
- `trace_id`: a `UUID` shared with the corresponding [`BeforeAutostartEvent`](@ref)

See also: [`BeforeAutostartEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct AfterAutostartEvent{E} <: Event{:after_autostart}
    engine::E
    trace_id::UUID
end
