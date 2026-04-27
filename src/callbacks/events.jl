export BeforeModelCreationEvent, AfterModelCreationEvent
export BeforeInferenceEvent, AfterInferenceEvent
export BeforeIterationEvent, AfterIterationEvent
export BeforeDataUpdateEvent, AfterDataUpdateEvent
export OnMarginalUpdateEvent
export BeforeAutostartEvent, AfterAutostartEvent

import ReactiveMP: Event, event_name, generate_span_id, getdata

## RxInfer-level callback event types
## These events subtype `ReactiveMP.Event{E}` and carry relevant data as fields.

"""
    BeforeModelCreationEvent{S} <: ReactiveMP.Event{:before_model_creation}

Fires right before the probabilistic model is created in the [`infer`](@ref) function.

# Fields
- `span_id`: an identifier shared with the corresponding [`AfterModelCreationEvent`](@ref)

See also: [`AfterModelCreationEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct BeforeModelCreationEvent{S} <: Event{:before_model_creation}
    span_id::S
end

"""
    AfterModelCreationEvent{M, S} <: ReactiveMP.Event{:after_model_creation}

Fires right after the probabilistic model is created in the [`infer`](@ref) function.

# Fields
- `model::M`: the created [`ProbabilisticModel`](@ref) instance
- `span_id`: an identifier shared with the corresponding [`BeforeModelCreationEvent`](@ref)

See also: [`BeforeModelCreationEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct AfterModelCreationEvent{M, S} <: Event{:after_model_creation}
    model::M
    span_id::S
end

"""
    BeforeInferenceEvent{M, S} <: ReactiveMP.Event{:before_inference}

Fires right before the inference procedure starts (after model creation and subscription setup).

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `span_id`: an identifier shared with the corresponding [`AfterInferenceEvent`](@ref)

See also: [`AfterInferenceEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct BeforeInferenceEvent{M, S} <: Event{:before_inference}
    model::M
    span_id::S
end

"""
    AfterInferenceEvent{M, S} <: ReactiveMP.Event{:after_inference}

Fires right after the inference procedure completes.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `span_id`: an identifier shared with the corresponding [`BeforeInferenceEvent`](@ref)

See also: [`BeforeInferenceEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct AfterInferenceEvent{M, S} <: Event{:after_inference}
    model::M
    span_id::S
end

"""
    BeforeIterationEvent{M, S} <: ReactiveMP.Event{:before_iteration}

Fires right before each variational iteration.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `iteration::Int`: the current iteration number
- `stop_iteration::Bool`: set to `true` from a callback to halt iterations early (default: `false`)
- `span_id`: an identifier shared with the corresponding [`AfterIterationEvent`](@ref)

See also: [`AfterIterationEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
mutable struct BeforeIterationEvent{M, S} <: Event{:before_iteration}
    model::M
    iteration::Int
    stop_iteration::Bool
    span_id::S
end

BeforeIterationEvent(model::M, iteration::Int, span_id) where {M} = BeforeIterationEvent(
    model, iteration, false, span_id
)

"""
    AfterIterationEvent{M, S} <: ReactiveMP.Event{:after_iteration}

Fires right after each variational iteration.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `iteration::Int`: the current iteration number
- `stop_iteration::Bool`: set to `true` from a callback to halt iterations early (default: `false`)
- `span_id`: an identifier shared with the corresponding [`BeforeIterationEvent`](@ref)

See also: [`BeforeIterationEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
mutable struct AfterIterationEvent{M, S} <: Event{:after_iteration}
    model::M
    iteration::Int
    stop_iteration::Bool
    span_id::S
end

AfterIterationEvent(model::M, iteration::Int, span_id) where {M} = AfterIterationEvent(
    model, iteration, false, span_id
)

"""
    BeforeDataUpdateEvent{M, D, S} <: ReactiveMP.Event{:before_data_update}

Fires right before updating data variables in each iteration.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `data::D`: the data being used for the update
- `span_id`: an identifier shared with the corresponding [`AfterDataUpdateEvent`](@ref)

See also: [`AfterDataUpdateEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct BeforeDataUpdateEvent{M, D, S} <: Event{:before_data_update}
    model::M
    data::D
    span_id::S
end

"""
    AfterDataUpdateEvent{M, D, S} <: ReactiveMP.Event{:after_data_update}

Fires right after updating data variables in each iteration.

# Fields
- `model::M`: the [`ProbabilisticModel`](@ref) instance
- `data::D`: the data that was used for the update
- `span_id`: an identifier shared with the corresponding [`BeforeDataUpdateEvent`](@ref)

See also: [`BeforeDataUpdateEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct AfterDataUpdateEvent{M, D, S} <: Event{:after_data_update}
    model::M
    data::D
    span_id::S
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
    BeforeAutostartEvent{E, S} <: ReactiveMP.Event{:before_autostart}

Fires right before `RxInfer.start()` is called on the streaming inference engine (when `autostart = true`).

# Fields
- `engine::E`: the [`RxInferenceEngine`](@ref) instance
- `span_id`: an identifier shared with the corresponding [`AfterAutostartEvent`](@ref)

See also: [`AfterAutostartEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct BeforeAutostartEvent{E, S} <: Event{:before_autostart}
    engine::E
    span_id::S
end

"""
    AfterAutostartEvent{E, S} <: ReactiveMP.Event{:after_autostart}

Fires right after `RxInfer.start()` is called on the streaming inference engine (when `autostart = true`).

# Fields
- `engine::E`: the [`RxInferenceEngine`](@ref) instance
- `span_id`: an identifier shared with the corresponding [`BeforeAutostartEvent`](@ref)

See also: [`BeforeAutostartEvent`](@ref), [Callbacks](@ref manual-inference-callbacks)
"""
struct AfterAutostartEvent{E, S} <: Event{:after_autostart}
    engine::E
    span_id::S
end

## Compact `Base.show` methods for Tier A callback events. Format: single-line
## `EventName(k=v, ...)`. Span identifiers are truncated to a 4-character prefix
## so trace output (e.g. TBLogger Text tab) stays readable.

# Render a span identifier as a short 4-character prefix.
function _show_span(io::IO, span_id)
    s = string(span_id)
    if length(s) >= 4
        print(io, SubString(s, 1, 4), "…")
    else
        print(io, s)
    end
end

_show_model(io::IO, model) = print(io, nameof(typeof(model)))

function Base.show(io::IO, ev::BeforeModelCreationEvent)
    print(io, "BeforeModelCreationEvent(span=")
    _show_span(io, ev.span_id)
    print(io, ")")
end

function Base.show(io::IO, ev::AfterModelCreationEvent)
    print(io, "AfterModelCreationEvent(model=")
    _show_model(io, ev.model)
    print(io, ", span=")
    _show_span(io, ev.span_id)
    print(io, ")")
end

function Base.show(io::IO, ev::BeforeInferenceEvent)
    print(io, "BeforeInferenceEvent(model=")
    _show_model(io, ev.model)
    print(io, ", span=")
    _show_span(io, ev.span_id)
    print(io, ")")
end

function Base.show(io::IO, ev::AfterInferenceEvent)
    print(io, "AfterInferenceEvent(model=")
    _show_model(io, ev.model)
    print(io, ", span=")
    _show_span(io, ev.span_id)
    print(io, ")")
end

function Base.show(io::IO, ev::BeforeIterationEvent)
    print(io, "BeforeIterationEvent(iter=", ev.iteration)
    ev.stop_iteration && print(io, ", stop=true")
    print(io, ", span=")
    _show_span(io, ev.span_id)
    print(io, ")")
end

function Base.show(io::IO, ev::AfterIterationEvent)
    print(io, "AfterIterationEvent(iter=", ev.iteration)
    ev.stop_iteration && print(io, ", stop=true")
    print(io, ", span=")
    _show_span(io, ev.span_id)
    print(io, ")")
end

function Base.show(io::IO, ev::BeforeDataUpdateEvent)
    print(io, "BeforeDataUpdateEvent(data=", collect(keys(ev.data)))
    print(io, ", span=")
    _show_span(io, ev.span_id)
    print(io, ")")
end

function Base.show(io::IO, ev::AfterDataUpdateEvent)
    print(io, "AfterDataUpdateEvent(data=", collect(keys(ev.data)))
    print(io, ", span=")
    _show_span(io, ev.span_id)
    print(io, ")")
end

function Base.show(io::IO, ev::OnMarginalUpdateEvent)
    print(io, "OnMarginalUpdateEvent(var=:", ev.variable_name, ", update=")
    show(io, getdata(ev.update))
    print(io, ")")
end

function Base.show(io::IO, ev::BeforeAutostartEvent)
    print(io, "BeforeAutostartEvent(engine=", nameof(typeof(ev.engine)), ", span=")
    _show_span(io, ev.span_id)
    print(io, ")")
end

function Base.show(io::IO, ev::AfterAutostartEvent)
    print(io, "AfterAutostartEvent(engine=", nameof(typeof(ev.engine)), ", span=")
    _show_span(io, ev.span_id)
    print(io, ")")
end
