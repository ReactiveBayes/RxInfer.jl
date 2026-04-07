export RxInferTraceCallbacks, TracedEvent

"""
    TracedEvent

A single traced event recorded by [`RxInferTraceCallbacks`](@ref).
Wraps the original event object (a subtype of `ReactiveMP.Event`).

# Fields
- `event::ReactiveMP.Event`: the event object that was passed to the callback
- `time_ns::UInt64`: the timestamp of the event in nanoseconds, uses `time_ns()` function from Julia


Use `ReactiveMP.event_name(traced_event.event)` to retrieve the event name as a `Symbol`.
"""
struct TracedEvent
    event::Event
    time_ns::UInt64
end

Base.summary(io::IO, te::TracedEvent) =
    print(io, "TracedEvent(:$(event_name(typeof(te.event))))")

"""
    RxInferTraceCallbacks()

A callback structure that records all callback events during the inference procedure.
Each event is stored as a [`TracedEvent`](@ref) wrapping the original event object.

After model creation, the trace callbacks instance is automatically saved into the model's metadata
under the `:trace` key (i.e., `model.metadata[:trace]`), making it accessible from the inference result via
`result.model.metadata[:trace]`.

Use `RxInfer.tracedevents(callbacks)` to retrieve the vector of traced events.

# Example
```julia
# Create a trace callbacks instance
trace = RxInferTraceCallbacks()

result = infer(
    model = my_model(),
    data = my_data,
    callbacks = trace,
)

# Access the traced events
events = RxInfer.tracedevents(trace)
for event in events
    println(event_name(event.event))
end

# Or access via model metadata
result.model.metadata[:trace] === trace # true
```
"""
struct RxInferTraceCallbacks
    events::Vector{TracedEvent}
end

RxInferTraceCallbacks() = RxInferTraceCallbacks(TracedEvent[])

"""
    tracedevents(callbacks::RxInferTraceCallbacks)

Returns the vector of [`TracedEvent`](@ref) recorded by the trace callbacks.

See also: [`RxInferTraceCallbacks`](@ref).
"""
tracedevents(callbacks::RxInferTraceCallbacks) = callbacks.events

"""
    tracedevents(event::Symbol, callbacks::RxInferTraceCallbacks)

Returns the vector of [`TracedEvent`](@ref) recorded by the trace callbacks filtered by `event`.

See also: [`RxInferTraceCallbacks`](@ref).
"""
tracedevents(event::Symbol, callbacks::RxInferTraceCallbacks) =
    filter(e -> event_name(typeof(e.event)) == event, callbacks.events)

Base.isempty(callbacks::RxInferTraceCallbacks) = isempty(callbacks.events)

function _event_name_to_type_name(name::Symbol)
    return join(map(uppercasefirst, split(string(name), "_"))) * "Event"
end

function Base.show(io::IO, callbacks::RxInferTraceCallbacks)
    if isempty(callbacks)
        print(io, "RxInferTraceCallbacks (empty, no events recorded)")
    else
        events = callbacks.events
        names = unique(event_name(typeof(e.event)) for e in events)
        println(
            io, "RxInferTraceCallbacks (", length(events), " events recorded)"
        )
        for name in names
            println(io, "  :$name ")
        end
        hint_event = _event_name_to_type_name(first(names))
        print(
            io,
            "Use `?",
            hint_event,
            " or ",
            "@doc(",
            hint_event,
            ")",
            "` to see the documentation for an event.",
        )
    end
end

import ReactiveMP: handle_event, Event, event_name

# Catch-all: trace every event
function ReactiveMP.handle_event(callbacks::RxInferTraceCallbacks, event::Event)
    push!(callbacks.events, TracedEvent(event, time_ns()))
    return nothing
end

# Special handling for :after_model_creation to save to metadata
function ReactiveMP.handle_event(
    callbacks::RxInferTraceCallbacks, event::AfterModelCreationEvent
)
    if haskey(event.model.metadata, :trace)
        error(
            "The model's metadata already contains a `:trace` key. " *
            "This can happen if you pass `trace = true` while also providing " *
            "`RxInferTraceCallbacks` in the `callbacks` argument. Use one or the other, not both.",
        )
    end
    event.model.metadata[:trace] = callbacks
    push!(callbacks.events, TracedEvent(event, time_ns()))
    return nothing
end

function convert_to_tensorboard end # This function is implemented in the external module `TensorBoardLoggerExt` to avoid adding a hard dependency on TensorBoardLogger.jl for users who don't need it.