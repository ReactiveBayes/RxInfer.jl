module PrettyTablesExt

using RxInfer, PrettyTables

using PrettyTables.Printf

function prettytime(t::Union{UInt64, Float64})
    if t < 1.0e3
        value, units = t, "ns"
    elseif t < 1.0e6
        value, units = t / 1.0e3, "μs"
    elseif t < 1.0e9
        value, units = t / 1.0e6, "ms"
    else
        value, units = t / 1.0e9, "s"
    end
    return string(@sprintf("%.3f", value), " ", units)
end

prettytime(s) = s

function PrettyTables.pretty_table(
        io::IO, callbacks::RxInferBenchmarkCallbacks; kwargs...
    )
    default_column_labels = ["Operation", "Min", "Max", "Mean", "Median", "Std"]
    default_highlighter = TextHighlighter(
        (data, i, j) -> (j == 3) && (data[i, j] > 10 * data[i, j - 1]),
        crayon"red bold",
    )

    print(
        io,
        "RxInfer inference benchmark statistics: $(length(callbacks.before_model_creation_ts)) evaluations \n",
    )
    return if !isempty(callbacks)
        benchmark_data = RxInfer.get_benchmark_stats(callbacks)
        pretty_table(
            io,
            benchmark_data;
            formatters = [(s, i, j) -> prettytime(s)],
            column_labels = default_column_labels,
            style = TextTableStyle(;
                first_line_column_label = crayon"yellow bold"
            ),
            table_format = TextTableFormat(;
                borders = text_table_borders__unicode_rounded
            ),
            highlighters = [default_highlighter],
            kwargs...,
        )
    end
end

# The tables of an inference report, through PrettyTables: rounded borders, the row groups as
# separators, warnings in red and notes in yellow
function RxInfer.render_report_table_with(::typeof(RxInfer.render_report_table), io::IO, table::RxInfer.ReportTable; kwargs...)
    isempty(table) && return nothing
    data = permutedims(reduce(hcat, table.rows))
    levels = table.levels
    highlighters = [
        TextHighlighter((data, i, j) -> levels[i] === :warning, crayon"red bold"),
        TextHighlighter((data, i, j) -> levels[i] === :note, crayon"yellow"),
    ]
    return pretty_table(
        io,
        data;
        title = table.title,
        column_labels = table.header,
        row_group_labels = table.groups,
        style = TextTableStyle(; first_line_column_label = crayon"yellow bold", row_group_label = crayon"bold"),
        table_format = TextTableFormat(; borders = text_table_borders__unicode_rounded),
        highlighters = highlighters,
        alignment = :l,
        fit_table_in_display_horizontally = false,
        fit_table_in_display_vertically = false,
        kwargs...,
    )
end

# `pretty_table(io, report)`, as for the benchmark callbacks: each section's summary, then its tables
function PrettyTables.pretty_table(io::IO, report::RxInfer.InferenceReport; kwargs...)
    print(io, "Inference report")
    for section in report.sections
        print(io, "\n  ", RxInfer.report_summary(section))
        for table in RxInfer.report_tables(section)
            println(io)
            RxInfer.render_report_table_with(RxInfer.render_report_table, io, table; kwargs...)
        end
        footer = RxInfer.report_footer(section)
        isnothing(footer) || print(io, "\n    ", footer)
    end
    return nothing
end

# Plug into the `summarize_invokes` function to use `pretty_table` if `PrettyTables.jl` is installed
function RxInfer.summarize_invokes_pretty_table(
        ::typeof(RxInfer.summarize_invokes), io::IO, data; kwargs...
    )
    return pretty_table(io, data; kwargs...)
end

end
