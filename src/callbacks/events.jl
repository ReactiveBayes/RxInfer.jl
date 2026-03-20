export BeforeModelCreationEvent, AfterModelCreationEvent
export BeforeInferenceEvent, AfterInferenceEvent
export BeforeIterationEvent, AfterIterationEvent
export BeforeDataUpdateEvent, AfterDataUpdateEvent
export OnMarginalUpdateEvent
export BeforeAutostartEvent, AfterAutostartEvent

import ReactiveMP: Event, event_name

## RxInfer-level callback event types
## These events subtype `ReactiveMP.Event{E}` and carry relevant data as fields.

"""
    BeforeModelCreationEvent <: ReactiveMP.Event{:before_model_creation}

Fires right before the probabilistic model is created in the [`infer`](@ref) function.

See also: [`AfterModelCreationEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct BeforeModelCreationEvent <: Event{:before_model_creation} end

"""
    AfterModelCreationEvent{M} <: ReactiveMP.Event{:after_model_creation}

Fires right after the probabilistic model is created in the [`infer`](@ref) function.

# Fields
- `model::M`: the created [`ProbabilisticModel`](@ref) instance

See also: [`BeforeModelCreationEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct AfterModelCreationEvent{M} <: Event{:after_model_creation}
    model::M
end

"""
    BeforeInferenceEvent{M} <: ReactiveMP.Event{:before_inference}

Fires right before the inference procedure starts (after model creation and subscription setup).

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance

See also: [`AfterInferenceEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct BeforeInferenceEvent{M} <: Event{:before_inference}
    model::M
end

"""
    AfterInferenceEvent{M} <: ReactiveMP.Event{:after_inference}

Fires right after the inference procedure completes.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance

See also: [`BeforeInferenceEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct AfterInferenceEvent{M} <: Event{:after_inference}
    model::M
end

"""
    BeforeIterationEvent{M} <: ReactiveMP.Event{:before_iteration}

Fires right before each variational iteration.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `iteration::Int`: the current iteration number

See also: [`AfterIterationEvent`](@ref), [`StopIteration`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct BeforeIterationEvent{M} <: Event{:before_iteration}
    model::M
    iteration::Int
end

"""
    AfterIterationEvent{M} <: ReactiveMP.Event{:after_iteration}

Fires right after each variational iteration.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `iteration::Int`: the current iteration number

See also: [`BeforeIterationEvent`](@ref), [`StopIteration`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct AfterIterationEvent{M} <: Event{:after_iteration}
    model::M
    iteration::Int
end

"""
    BeforeDataUpdateEvent{M, D} <: ReactiveMP.Event{:before_data_update}

Fires right before updating data variables in each iteration.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `data::D`: the data being used for the update

See also: [`AfterDataUpdateEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct BeforeDataUpdateEvent{M, D} <: Event{:before_data_update}
    model::M
    data::D
end

"""
    AfterDataUpdateEvent{M, D} <: ReactiveMP.Event{:after_data_update}

Fires right after updating data variables in each iteration.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `data::D`: the data that was used for the update

See also: [`BeforeDataUpdateEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct AfterDataUpdateEvent{M, D} <: Event{:after_data_update}
    model::M
    data::D
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

See also: [`AfterAutostartEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct BeforeAutostartEvent{E} <: Event{:before_autostart}
    engine::E
end

"""
    AfterAutostartEvent{E} <: ReactiveMP.Event{:after_autostart}

Fires right after `RxInfer.start()` is called on the streaming inference engine (when `autostart = true`).

# Fields
- `engine::E`: the [`RxInferenceEngine`](@ref) instance

See also: [`BeforeAutostartEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct AfterAutostartEvent{E} <: Event{:after_autostart}
    engine::E
end
