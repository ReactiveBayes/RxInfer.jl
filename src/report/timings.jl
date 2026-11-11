export TimingsReport

import ReactiveMP:
    BeforeMessageRuleCallEvent,
    BeforeMarginalRuleCallEvent,
    BeforeFactorBoundFreeEnergyEvent,
    BeforeFormConstraintAppliedEvent

"""
    TimingsReport

The section of an [`InferenceReport`](@ref) that says where the time of one inference went,
collected from the same events as the approximations, `report = (:timings,)`:

- `model`, the nanoseconds the model's creation took, and `inference`, the inference's;
- `iterations`, the nanoseconds of each iteration;
- `kinds`, for each kind of work, `:rules` (exact rules), `:fallback` (the rule fallback),
  `:projections` (form constraints) and `:free_energy` (its terms), the nanoseconds spent and the
  number of calls, `(time, calls)`; each call's time is its own, what it called is counted where
  it belongs, so the kinds add up;
- `nodes`, the same by `(node type, kind)`, and `variables`, the projections' by variable name.

What is left of the inference's time went to the engine itself: the products of messages and the
streams. For statistics over many runs, use [`RxInferBenchmarkCallbacks`](@ref).
"""
mutable struct TimingsReport
    model::UInt64
    inference::UInt64
    iterations::Vector{UInt64}
    kinds::Dict{Symbol, Tuple{UInt64, Int}}
    nodes::Dict{Tuple{Any, Symbol}, Tuple{UInt64, Int}}
    variables::Dict{Symbol, Tuple{UInt64, Int}}
    # the spans open now, innermost last: their start and the time their children took
    open::Vector{Tuple{Any, UInt64, UInt64}}
    starts::Dict{Symbol, UInt64}
end

TimingsReport() = TimingsReport(0, 0, UInt64[], Dict(), Dict(), Dict(), Tuple{Any, UInt64, UInt64}[], Dict())

const TIMED_BEFORE = Union{BeforeMessageRuleCallEvent, BeforeMarginalRuleCallEvent, BeforeFactorBoundFreeEnergyEvent, BeforeFormConstraintAppliedEvent}
const TIMED_AFTER = Union{AfterMessageRuleCallEvent, AfterMarginalRuleCallEvent, AfterFactorBoundFreeEnergyEvent, AfterFormConstraintAppliedEvent}
const TIMED_STAGES = Union{BeforeModelCreationEvent, AfterModelCreationEvent, BeforeInferenceEvent, AfterInferenceEvent, BeforeIterationEvent, AfterIterationEvent}

ReactiveMP.listens(::TimingsReport, ::Type{T}) where {T} = T <: Union{TIMED_BEFORE, TIMED_AFTER, TIMED_STAGES}

record!(report::TimingsReport, event::TIMED_BEFORE) = (push!(report.open, (event.span_id, time_ns(), UInt64(0))); nothing)

function record!(report::TimingsReport, event::TIMED_AFTER)
    now = time_ns()
    # the matching span, innermost; one left open by an error is dropped
    i = findlast(span -> span[1] == event.span_id, report.open)
    isnothing(i) && return nothing
    _, start, children = report.open[i]
    resize!(report.open, i - 1)
    elapsed = now - start
    own = elapsed - min(children, elapsed)
    if !isempty(report.open)
        span, parentstart, parentchildren = report.open[end]
        report.open[end] = (span, parentstart, parentchildren + elapsed)
    end
    kind = timed_kind(event)
    isnothing(kind) && return nothing
    add_time!(report.kinds, kind, own)
    if event isa AfterFormConstraintAppliedEvent
        add_time!(report.variables, first(variable_key(event.variable.label)), own)
    else
        add_time!(report.nodes, (event_node(event), kind), own)
    end
    return nothing
end

function add_time!(times, key, ns)
    t, n = get(times, key, (UInt64(0), 0))
    times[key] = (t + ns, n + 1)
    return times
end

timed_kind(event::Union{AfterMessageRuleCallEvent, AfterMarginalRuleCallEvent}) =
    isnothing(event.rule) ? nothing : event.rule isa MessagePassingRulesBase.RuleSpec ? :rules : :fallback
timed_kind(::AfterFactorBoundFreeEnergyEvent) = :free_energy
timed_kind(::AfterFormConstraintAppliedEvent) = :projections

record!(report::TimingsReport, ::BeforeModelCreationEvent) = (report.starts[:model] = time_ns(); nothing)
record!(report::TimingsReport, ::AfterModelCreationEvent) = (report.model += time_ns() - get(report.starts, :model, time_ns()); nothing)
record!(report::TimingsReport, ::BeforeInferenceEvent) = (report.starts[:inference] = time_ns(); nothing)
record!(report::TimingsReport, ::AfterInferenceEvent) = (report.inference += time_ns() - get(report.starts, :inference, time_ns()); nothing)
record!(report::TimingsReport, ::BeforeIterationEvent) = (report.starts[:iteration] = time_ns(); nothing)
record!(report::TimingsReport, ::AfterIterationEvent) = (push!(report.iterations, time_ns() - get(report.starts, :iteration, time_ns())); nothing)

const TIMED_KIND_NAMES = (rules = "exact rules", fallback = "rule fallback", projections = "projections", free_energy = "free energy terms")

function report_summary(report::TimingsReport)
    n = length(report.iterations)
    text = string("Timings: model ", time_display(report.model))
    report.inference > 0 && (text *= string(", inference ", time_display(report.inference)))
    n > 0 && (text *= string(" (", n, n == 1 ? " iteration" : " iterations", ")"))
    return text
end

function report_tables(report::TimingsReport)
    total = report.inference > 0 ? report.inference : sum(first, values(report.kinds); init = UInt64(0))
    share(ns) = total > 0 ? string(round(100 * ns / total; digits = 1), "%") : ""
    rows = Vector{String}[]
    for kind in keys(TIMED_KIND_NAMES)
        haskey(report.kinds, kind) || continue
        ns, calls = report.kinds[kind]
        push!(rows, [TIMED_KIND_NAMES[kind], time_display(ns), share(ns), string(calls)])
    end
    if report.inference > 0
        rest = report.inference - min(report.inference, sum(first, values(report.kinds); init = UInt64(0)))
        push!(rows, ["the engine, products and streams", time_display(rest), share(rest), ""])
    end
    bykind = ReportTable("Time by kind", ["", "time", "of the inference", "calls"], rows)
    entries = [(node_function_name(node), TIMED_KIND_NAMES[kind], ns, calls) for ((node, kind), (ns, calls)) in report.nodes]
    append!(entries, [(variables_display([(name, nothing)]), "projections", ns, calls) for (name, (ns, calls)) in report.variables])
    sort!(entries; by = e -> -Int(e[3]))
    top = [[name, kind, time_display(ns), share(ns), string(calls)] for (name, kind, ns, calls) in entries[1:min(end, 8)]]
    bynode = ReportTable(length(entries) > 8 ? "The most time, by node and variable (8 of $(length(entries)))" : "Time by node and variable", ["", "kind", "time", "of the inference", "calls"], top)
    return [bykind, bynode]
end

Base.show(io::IO, ::MIME"text/plain", report::TimingsReport) = show_report_section(io, report)
