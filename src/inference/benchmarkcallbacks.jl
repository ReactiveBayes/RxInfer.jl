using PrettyTables
using PrettyTables.Printf

export RxInferBenchmarkCallbacks

"""
    RxInferBenchmarkCallbacks

A callback structure for collecting timing information during the inference procedure.
This structure collects timestamps for various stages of the inference process and aggregates
them across multiple runs, allowing you to track performance statistics (min/max/average/etc.)
of your model's creation and inference procedure. The structure supports pretty printing by default,
displaying timing statistics in a human-readable format.

# Fields
- `before_model_creation_ts`: Vector of timestamps before model creation
- `after_model_creation_ts`: Vector of timestamps after model creation
- `before_inference_ts`: Vector of timestamps before inference starts
- `after_inference_ts`: Vector of timestamps after inference ends
- `before_iteration_ts`: Vector of vectors of timestamps before each iteration
- `after_iteration_ts`: Vector of vectors of timestamps after each iteration
- `before_autostart_ts`: Vector of timestamps before autostart
- `after_autostart_ts`: Vector of timestamps after autostart

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
    before_model_creation_ts::Vector{UInt64}
    after_model_creation_ts::Vector{UInt64}
    before_inference_ts::Vector{UInt64}
    after_inference_ts::Vector{UInt64}
    before_iteration_ts::Vector{Vector{UInt64}}
    after_iteration_ts::Vector{Vector{UInt64}}
    before_autostart_ts::Vector{UInt64}
    after_autostart_ts::Vector{UInt64}
end

RxInferBenchmarkCallbacks() = RxInferBenchmarkCallbacks(UInt64[], UInt64[], UInt64[], UInt64[], Vector{UInt64}[], Vector{UInt64}[], UInt64[], UInt64[])

check_available_callbacks(warn, callbacks::RxInferBenchmarkCallbacks, ::Val{AvailableCallbacks}) where {AvailableCallbacks} = nothing

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

inference_get_callback(callbacks::RxInferBenchmarkCallbacks, name::Symbol) = nothing

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

function __get_benchmark_stats(callbacks::RxInferBenchmarkCallbacks)
    model_creation_time = callbacks.after_model_creation_ts .- callbacks.before_model_creation_ts
    stats_to_show = [("Model creation", model_creation_time)]
    inference_time = callbacks.after_inference_ts .- callbacks.before_inference_ts
    iteration_time = [callbacks.after_iteration_ts[i] .- callbacks.before_iteration_ts[i] for i in 1:length(callbacks.before_iteration_ts)]
    if length(inference_time) > 0
        push!(stats_to_show, ("Inference", inference_time))
        push!(stats_to_show, ("Iteration", reshape(stack(iteration_time), :)))
    end
    autostart_time = callbacks.after_autostart_ts .- callbacks.before_autostart_ts
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

    data = __get_benchmark_stats(callbacks)
    hl_v = Highlighter((data, i, j) -> (j == 3) && (data[i, j] > 10 * data[i, j - 1]), crayon"red bold")
    pretty_table(io, data; formatters = (s, i, j) -> prettytime(s), header = header, header_crayon = crayon"yellow bold", tf = tf_unicode_rounded, highlighters = hl_v)
end
