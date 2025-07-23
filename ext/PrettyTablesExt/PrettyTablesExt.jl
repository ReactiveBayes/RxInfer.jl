module PrettyTablesExt

using RxInfer, PrettyTables

using PrettyTables.Printf

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

function PrettyTables.pretty_table(io::IO, callbacks::RxInferBenchmarkCallbacks; kwargs...)
    default_header = (["Operation", "Min", "Max", "Mean", "Median", "Std"],)
    default_highlighter = Highlighter((data, i, j) -> (j == 3) && (data[i, j] > 10 * data[i, j - 1]), crayon"red bold")

    print(io, "RxInfer inference benchmark statistics: $(length(callbacks.before_model_creation_ts)) evaluations \n")
    if !isempty(callbacks)
        benchmark_data = RxInfer.get_benchmark_stats(callbacks)
        pretty_table(
            io,
            benchmark_data;
            formatters = (s, i, j) -> prettytime(s),
            header = default_header,
            header_crayon = crayon"yellow bold",
            tf = tf_unicode_rounded,
            highlighters = default_highlighter,
            kwargs...
        )
    end
end

# Plugin into `summarize_invokes` function to use `pretty_table` if `PrettyTables.jl` is installed
function RxInfer.summarize_invokes_pretty_table(::typeof(RxInfer.summarize_invokes), io::IO, data; kwargs...)
    return pretty_table(io, data; kwargs...)
end

end
