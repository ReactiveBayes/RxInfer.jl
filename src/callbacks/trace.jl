export RxInferTraceCallbacks, TracedEvent

"""
    TracedEvent

A single traced event recorded by [`RxInferTraceCallbacks`](@ref).

# Fields
- `event::Symbol`: the name of the event (e.g. `:before_iteration`, `:after_model_creation`)
- `arguments::Tuple`: the arguments passed to the callback for this event
"""
struct TracedEvent
    event::Symbol
    arguments::Tuple
end

Base.show(io::IO, te::TracedEvent) =
    print(io, "TracedEvent(:$(te.event), $(length(te.arguments)) args)")

"""
    RxInferTraceCallbacks()

A callback structure that records all callback events during the inference procedure.
Each event is stored as a [`TracedEvent`](@ref) containing the event name and its arguments.

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
    println(event.event, " with ", length(event.arguments), " arguments")
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
    filter(e -> e.event == event, callbacks.events)

Base.isempty(callbacks::RxInferTraceCallbacks) = isempty(callbacks.events)

function Base.show(io::IO, callbacks::RxInferTraceCallbacks)
    if isempty(callbacks)
        print(io, "RxInferTraceCallbacks (empty, no events recorded)")
    else
        print(
            io,
            "RxInferTraceCallbacks (",
            length(callbacks.events),
            " events recorded)",
        )
    end
end

import ReactiveMP: invoke_callback

# Catch-all: trace every event
function ReactiveMP.invoke_callback(
    callbacks::RxInferTraceCallbacks, event::Val{E}, args...
) where {E}
    push!(callbacks.events, TracedEvent(E, args))
    return nothing
end

# Special handling for :after_model_creation to save to metadata
function ReactiveMP.invoke_callback(
    callbacks::RxInferTraceCallbacks,
    ::Val{:after_model_creation},
    model,
    args...,
)
    if haskey(model.metadata, :trace)
        error(
            "The model's metadata already contains a `:trace` key. " *
            "This can happen if you pass `trace = true` while also providing " *
            "`RxInferTraceCallbacks` in the `callbacks` argument. Use one or the other, not both.",
        )
    end
    model.metadata[:trace] = callbacks
    push!(
        callbacks.events, TracedEvent(:after_model_creation, (model, args...))
    )
    return nothing
end
