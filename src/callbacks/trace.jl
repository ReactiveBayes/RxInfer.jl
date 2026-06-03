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

TracedEvent(event::Event) = TracedEvent(event, time_ns())

Base.show(io::IO, te::TracedEvent) =
    print(io, "TracedEvent(:$(event_name(typeof(te.event))))")

"""
    RxInferTraceCallbacks()

A callback structure that records (optionally filtered) callback events during the inference procedure.
Each event is stored as a [`TracedEvent`](@ref) wrapping the original event object.

When constructed with no arguments (or `trace = true`), all events are recorded.
When constructed with a tuple of `Symbol`s, only events whose names are in that tuple are recorded.

After model creation, the trace callbacks instance is automatically saved into the model's metadata
under the `:trace` key (i.e., `model.metadata[:trace]`), making it accessible from the inference result via
`result.model.metadata[:trace]`.

Use `RxInfer.tracedevents(callbacks)` to retrieve the vector of traced events.

# Example
```julia
# Create a trace callbacks instance that records all events
trace = RxInferTraceCallbacks()

# Or record only specific events
trace = RxInferTraceCallbacks((:before_iteration, :after_iteration))

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
    include::Union{Nothing, Set{Symbol}}
end

RxInferTraceCallbacks() = RxInferTraceCallbacks(TracedEvent[], nothing)
RxInferTraceCallbacks(include::NTuple{N, Symbol}) where {N} =
    RxInferTraceCallbacks(TracedEvent[], Set{Symbol}(include))

"""
    is_trace_event_included(callbacks::RxInferTraceCallbacks, event_name::Symbol)

Checks whether the specified event is not filtered and should be traced.

```jldoctest
julia> callbacks = RxInfer.RxInferTraceCallbacks((:event1, :event2));

julia> RxInfer.is_trace_event_included(callbacks, :event1)
true

julia> RxInfer.is_trace_event_included(callbacks, :event2)
true

julia> RxInfer.is_trace_event_included(callbacks, :event3)
false
```
"""
function is_trace_event_included(
    callbacks::RxInferTraceCallbacks, event_name::Symbol
)
    if isnothing(callbacks.include)
        return true
    else
        return event_name ∈ callbacks.include
    end
end

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

# Catch-all: trace every event (respect the include filter)
function ReactiveMP.handle_event(callbacks::RxInferTraceCallbacks, event::Event)
    if is_trace_event_included(callbacks, event_name(event))
        push!(callbacks.events, TracedEvent(event))
    end
    return nothing
end

# Special handling for :after_model_creation to save to metadata
function ReactiveMP.handle_event(
    callbacks::RxInferTraceCallbacks, event::AfterModelCreationEvent
)
    if haskey(event.model.metadata, :trace)
        error(
            "The model's metadata already contains a `:trace` key. " *
            "This can happen if you pass `trace = true` (or a tuple of event names) while also providing " *
            "`RxInferTraceCallbacks` in the `callbacks` argument. Use one or the other, not both.",
        )
    end
    event.model.metadata[:trace] = callbacks
    if is_trace_event_included(callbacks, event_name(event))
        push!(callbacks.events, TracedEvent(event))
    end
    return nothing
end

"""
    convert_to_tensorboard(trace::RxInferTraceCallbacks; output_file::Union{String, Nothing} = nothing,
                           log_distributions::Bool = false, log_text_events::Bool = false,
                           n_samples::Int = 1024, verbose::Bool = true)

Convert trace events from inference to proper TensorFlow event files. Note that this function will not work 
unless `TensorBoardLogger.jl` is loaded in the present Julia session.

# Arguments
- `trace::RxInferTraceCallbacks`: The trace callbacks object from inference results
- `output_file::Union{String, Nothing}`: Optional directory path to write TensorBoard event logs. If not provided, writes to `tensorboard_logs/` in the current working directory.
- `log_distributions::Bool`: When `true`, log each univariate Normal and Gamma posterior as a per-iteration `HistogramSummary` so TensorBoard's **Distributions** tab renders a percentile-band view of the posterior across iterations. The same tag also appears in the **Histograms** tab as an offset ridgeline. Defaults to `false`.
- `log_text_events::Bool`: When `true`, emit a per-event text breadcrumb (e.g. `before_iteration`, `after_marginal_computation`, and the `Events` step timeline) into the **Text** tab. The `EventCounts` summary is always written regardless of this flag. Scalar and histogram outputs are unaffected. Defaults to `false`.
- `n_samples::Int`: Number of samples drawn from each posterior to build the per-iteration histogram when `log_distributions=true`. Defaults to 1024.
- `verbose`: Whether to print useful information during export or not, defaults to `true`

# Returns
- `String`: Path to the directory containing the TensorBoard event log files

# Description
This function processes all traced events and creates proper TensorFlow event files using TensorBoardLogger, which can be directly imported and visualized in TensorBoard. Outputs include:
- Text summaries with event type information and counts
- Scalar time-series for univariate Normal (`mean`, `precision`) and Gamma (`shape`, `rate`) posteriors
- Scalar time-series for per-iteration wall-clock duration (`iteration_time_ms`)
- When `log_distributions=true`: per-iteration `HistogramSummary` under `posteriors/<var>/distribution`, rendered primarily in TensorBoard's Distributions tab

The output directory can be directly opened in TensorBoard's web interface for visualization and analysis.

# Example
```julia
results = infer(
    model = my_model(),
    data = my_data,
    trace = true
)

trace = results.model.metadata[:trace]

# Create TensorBoard logs (writes to tensorboard_logs/ in the current directory)
log_dir = convert_to_tensorboard(trace; log_distributions = true)

# Then run: tensorboard --logdir=\$log_dir
```
"""
function convert_to_tensorboard(args...)
    # This function is implemented in the external module `TensorBoardLoggerExt` to avoid adding a hard dependency on TensorBoardLogger.jl for users who don't need it.
    error(
        "`convert_to_tensorboard` method with the specified arguments were not found. Did you load the `TensorBoardLogger.jl` in the current session? Otherwise consult the documentation.",
    )
end
