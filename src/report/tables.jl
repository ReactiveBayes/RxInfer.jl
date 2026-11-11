export ReportTable, report_tables

"""
    ReportTable(title, header, rows; groups = Pair{Int, String}[], levels = fill(:none, length(rows)))

A table of an inference report, plain strings ready to print or to read: a `title`, the column
`header`, the `rows`, each a vector of strings as long as the header, the row groups, `groups`, as
`first_row => label` pairs, and a `level` for each row, `:warning`, `:note` or `:none`, which a
display may highlight.

[`report_tables`](@ref) gives a section's tables; [`RxInfer.render_report_table`](@ref) prints one,
as aligned text, or with PrettyTables.jl when it is loaded.
"""
struct ReportTable
    title::String
    header::Vector{String}
    rows::Vector{Vector{String}}
    groups::Vector{Pair{Int, String}}
    levels::Vector{Symbol}
    function ReportTable(title, header, rows; groups = Pair{Int, String}[], levels = fill(:none, length(rows)))
        all(row -> length(row) == length(header), rows) || throw(ArgumentError("every row of a report table has as many cells as its header, $(length(header))"))
        length(levels) == length(rows) || throw(ArgumentError("a report table has a level for each row"))
        return new(String(title), collect(String, header), [collect(String, row) for row in rows], collect(Pair{Int, String}, groups), collect(Symbol, levels))
    end
end

Base.isempty(table::ReportTable) = isempty(table.rows)

"""
    report_tables(section) -> Vector{ReportTable}

The tables a section of an [`InferenceReport`](@ref) is shown as, plain data: for the
[`ApproximationsReport`](@ref), the warnings, the approximated posteriors with the messages that
formed them, and the joints and energies the rule fallback computed; for the
[`TimingsReport`](@ref), where the time went. A table with nothing to say is left out.
"""
function report_tables end

"""
    RxInfer.report_summary(section) -> String

The line a section of an [`InferenceReport`](@ref) starts with, such as
`"Approximations: 1 of 3 posteriors approximated, no warnings"`.
"""
function report_summary end

"""
    RxInfer.render_report_table(io::IO, table::ReportTable)

Print `table`: with PrettyTables.jl when it is loaded, rounded borders, row groups and warnings in
red, and as aligned text otherwise. The display of an [`InferenceReport`](@ref) prints its tables
with it.
"""
render_report_table(io::IO, table::ReportTable) = render_report_table_with(render_report_table, io, table)

# The plain rendering; `PrettyTablesExt` adds a more specific method for `render_report_table`, as
# it does for `summarize_invokes`.
function render_report_table_with(::Any, io::IO, table::ReportTable)
    isempty(table) && return nothing
    indent = "    "
    println(io, indent, table.title)
    widths = [maximum(textwidth, (table.header[j], (row[j] for row in table.rows)...)) for j in eachindex(table.header)]
    line(cells) = rstrip(join((rpad(cell, widths[j]) for (j, cell) in enumerate(cells)), "  "))
    println(io, indent, "  ", line(table.header))
    groups = Dict(table.groups)
    for (i, row) in enumerate(table.rows)
        haskey(groups, i) && println(io, indent, "  ", groups[i])
        marker = table.levels[i] === :warning ? "! " : table.levels[i] === :note ? "· " : "  "
        println(io, indent, isempty(groups) ? marker : string("  ", marker), line(row))
    end
    return nothing
end

# A line a section ends with, `nothing` for none
report_footer(section) = nothing

# A section's display: its summary, its tables, its footer
function show_report_section(io::IO, section)
    print(io, "  ", report_summary(section))
    for table in report_tables(section)
        isempty(table) && continue
        println(io)
        render_report_table(io, table)
    end
    footer = report_footer(section)
    isnothing(footer) || print(io, "\n    ", footer)
    return nothing
end

# Plain formatting shared by the sections

# Integers as compact ranges, `1:4`, `1:3, 7`, or their count where they are many
function index_display(indices)
    values = sort!(unique(indices))
    all(v -> v isa Integer, values) || return length(values) == 1 ? string(only(values)) : "…"
    runs = Vector{UnitRange{Int}}()
    for v in values
        if !isempty(runs) && last(runs[end]) + 1 == v
            runs[end] = first(runs[end]):v
        else
            push!(runs, v:v)
        end
    end
    length(runs) > 3 && return string("…(", length(values), ")")
    return join((length(r) == 1 ? string(first(r)) : string(first(r), ":", last(r)) for r in runs), ", ")
end

function time_display(ns::Real)
    ns < 1.0e3 && return string(round(Int, ns), " ns")
    ns < 1.0e6 && return string(round(ns / 1.0e3; sigdigits = 3), " μs")
    ns < 1.0e9 && return string(round(ns / 1.0e6; sigdigits = 3), " ms")
    return string(round(ns / 1.0e9; sigdigits = 3), " s")
end
