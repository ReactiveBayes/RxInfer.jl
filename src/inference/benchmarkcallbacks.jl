using PrettyTables
using PrettyTables.Printf
using DataStructures: CircularBuffer

export RxInferBenchmarkCallbacks

"""
    DEFAULT_BENCHMARK_CALLBACKS_BUFFER_CAPACITY

The default capacity of the circular buffers used to store timestamps in the `RxInferBenchmarkCallbacks` structure.
"""
const DEFAULT_BENCHMARK_CALLBACKS_BUFFER_CAPACITY = 1000

"""
    RxInferBenchmarkCallbacks(; capacity = RxInfer.DEFAULT_BENCHMARK_CALLBACKS_BUFFER_CAPACITY)

A callback structure for collecting timing information during the inference procedure.
This structure collects timestamps for various stages of the inference process and aggregates
them across multiple runs, allowing you to track performance statistics (min/max/average/etc.)
of your model's creation and inference procedure. The structure supports pretty printing by default,
displaying timing statistics in a human-readable format.

The structure uses circular buffers with a default capacity of $(DEFAULT_BENCHMARK_CALLBACKS_BUFFER_CAPACITY) entries to store timestamps,
which helps to limit memory usage in long-running applications. Use `RxInferBenchmarkCallbacks(; capacity = N)` to change the buffer capacity.
See also [`RxInfer.get_benchmark_stats(callbacks)`](@ref).

# Fields
- `before_model_creation_ts`: CircularBuffer of timestamps before model creation
- `after_model_creation_ts`: CircularBuffer of timestamps after model creation
- `before_inference_ts`: CircularBuffer of timestamps before inference starts
- `after_inference_ts`: CircularBuffer of timestamps after inference ends
- `before_iteration_ts`: CircularBuffer of vectors of timestamps before each iteration
- `after_iteration_ts`: CircularBuffer of vectors of timestamps after each iteration
- `before_autostart_ts`: CircularBuffer of timestamps before autostart
- `after_autostart_ts`: CircularBuffer of timestamps after autostart

# Example
```julia
# Create a callbacks instance to track performance
callbacks = RxInferBenchmarkCallbacks()

# Run inference multiple times to gather statistics
for _ in 1:10
    infer(
        model = my_model(),
        data = my_data,
        callbacks = callbacks
    )
end

# Display the timing statistics (uses pretty printing by default)
callbacks
```
"""
struct RxInferBenchmarkCallbacks
    before_model_creation_ts::CircularBuffer{UInt64}
    after_model_creation_ts::CircularBuffer{UInt64}
    before_inference_ts::CircularBuffer{UInt64}
    after_inference_ts::CircularBuffer{UInt64}
    before_iteration_ts::CircularBuffer{Vector{UInt64}}
    after_iteration_ts::CircularBuffer{Vector{UInt64}}
    before_autostart_ts::CircularBuffer{UInt64}
    after_autostart_ts::CircularBuffer{UInt64}
end

function RxInferBenchmarkCallbacks(; capacity = DEFAULT_BENCHMARK_CALLBACKS_BUFFER_CAPACITY)
    RxInferBenchmarkCallbacks(
        CircularBuffer{UInt64}(capacity),
        CircularBuffer{UInt64}(capacity),
        CircularBuffer{UInt64}(capacity),
        CircularBuffer{UInt64}(capacity),
        CircularBuffer{Vector{UInt64}}(capacity),
        CircularBuffer{Vector{UInt64}}(capacity),
        CircularBuffer{UInt64}(capacity),
        CircularBuffer{UInt64}(capacity)
    )
end

check_available_callbacks(warn, callbacks::RxInferBenchmarkCallbacks, ::Val{AvailableCallbacks}) where {AvailableCallbacks} = nothing
inference_get_callback(callbacks::RxInferBenchmarkCallbacks, name::Symbol) = nothing

Base.isempty(callbacks::RxInferBenchmarkCallbacks) = isempty(callbacks.before_model_creation_ts)

function inference_invoke_callback(callbacks::RxInferBenchmarkCallbacks, name::Symbol, args...)
    if name === :before_model_creation
        push!(callbacks.before_model_creation_ts, time_ns())
        push!(callbacks.before_iteration_ts, UInt64[])
        push!(callbacks.after_iteration_ts, UInt64[])
    elseif name === :after_model_creation
        push!(callbacks.after_model_creation_ts, time_ns())
    elseif name === :before_inference
        push!(callbacks.before_inference_ts, time_ns())
    elseif name === :after_inference
        push!(callbacks.after_inference_ts, time_ns())
    elseif name === :before_iteration
        push!(last(callbacks.before_iteration_ts), time_ns())
    elseif name === :after_iteration
        push!(last(callbacks.after_iteration_ts), time_ns())
    elseif name === :before_autostart
        push!(callbacks.before_autostart_ts, time_ns())
    elseif name === :after_autostart
        push!(callbacks.after_autostart_ts, time_ns())
    end
end

function prettytime(t::Union{UInt64, Float64})
    if t < 1e3
        value, units = t, "ns"
    elseif t < 1e6
        value, units = t / 1e3, "μs"
    elseif t < 1e9
        value, units = t / 1e6, "ms"
    else
        value, units = t / 1e9, "s"
    end
    return string(@sprintf("%.3f", value), " ", units)
end

prettytime(s) = s

"""
    get_benchmark_stats(callbacks::RxInferBenchmarkCallbacks)

Returns a matrix containing benchmark statistics for different operations in the inference process.
The matrix contains the following columns:
1. Operation name (String)
2. Minimum time (Float64)
3. Maximum time (Float64)
4. Mean time (Float64)
5. Median time (Float64)
6. Standard deviation (Float64)

Each row represents a different operation (model creation, inference, iteration, autostart).
Times are in nanoseconds.
"""
function get_benchmark_stats(callbacks::RxInferBenchmarkCallbacks)
    model_creation_time = collect(callbacks.after_model_creation_ts) .- collect(callbacks.before_model_creation_ts)
    stats_to_show = [("Model creation", model_creation_time)]
    inference_time = collect(callbacks.after_inference_ts) .- collect(callbacks.before_inference_ts)
    iteration_time = [collect(callbacks.after_iteration_ts[i]) .- collect(callbacks.before_iteration_ts[i]) for i in 1:length(callbacks.before_iteration_ts)]
    if length(inference_time) > 0
        push!(stats_to_show, ("Inference", inference_time))
        push!(stats_to_show, ("Iteration", reshape(stack(iteration_time), :)))
    end
    autostart_time = collect(callbacks.after_autostart_ts) .- collect(callbacks.before_autostart_ts)
    if length(autostart_time) > 0
        push!(stats_to_show, ("Autostart", autostart_time))
    end

    data = Matrix{Union{String, Float64}}(undef, length(stats_to_show), 6)

    for (i, (name, time)) in enumerate(stats_to_show)
        data[i, 1] = name
        data[i, 2] = convert(Float64, minimum(time))
        data[i, 3] = convert(Float64, maximum(time))
        data[i, 4] = convert(Float64, mean(time))
        data[i, 5] = convert(Float64, median(time))
        data[i, 6] = convert(Float64, std(time))
    end
    return data
end

function Base.show(io::IO, callbacks::RxInferBenchmarkCallbacks)
    if isempty(callbacks)
        return nothing
    end

    header = (["Operation", "Min", "Max", "Mean", "Median", "Std"],)

    print(io, "RxInfer inference benchmark statistics: $(length(callbacks.before_model_creation_ts)) evaluations \n")

    data = get_benchmark_stats(callbacks)
    hl_v = Highlighter((data, i, j) -> (j == 3) && (data[i, j] > 10 * data[i, j - 1]), crayon"red bold")
    pretty_table(io, data; formatters = (s, i, j) -> prettytime(s), header = header, header_crayon = crayon"yellow bold", tf = tf_unicode_rounded, highlighters = hl_v)
end
