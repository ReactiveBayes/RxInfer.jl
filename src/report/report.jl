export InferenceReport

"""
    InferenceReport

What `infer(...; report = true)` records about an inference, a section per subject, each read as a
property:
- `result.report.approximations`, the [`ApproximationsReport`](@ref): what was computed exactly
  and what was approximated, and how;
- `result.report.timings`, the [`TimingsReport`](@ref): where the time went.

`report` names the sections to record, `true` for all of them: `report = (:approximations,)`.
[`RxInfer.diagnostics`](@ref) lists the parts most likely to be inaccurate, and
[`report_tables`](@ref) gives each section's tables, which the display prints, with
PrettyTables.jl when it is loaded.

The report is collected from the engine's events, and costs nothing when off, the default. Read it
from the result of a batch inference, `result.report`, and from the engine of a streaming one,
`engine.model.metadata[:report]`.
"""
struct InferenceReport{S <: NamedTuple}
    sections::S
end

const REPORT_SECTIONS = (:approximations, :timings)

new_report_section(::Val{:approximations}) = ApproximationsReport()
new_report_section(::Val{:timings}) = TimingsReport()

function InferenceReport(names::Tuple)
    for name in names
        name in REPORT_SECTIONS || throw(
            ArgumentError(
                "`$(repr(name))` is not a section of the inference report; the sections are $(REPORT_SECTIONS)",
            ),
        )
    end
    return InferenceReport(NamedTuple{names}(map(name -> new_report_section(Val(name)), names)))
end

InferenceReport() = InferenceReport(REPORT_SECTIONS)

inference_report(report::Bool) = report ? InferenceReport() : nothing
inference_report(report::Tuple) = InferenceReport(report)
inference_report(report::Symbol) = InferenceReport((report,))

Base.getproperty(report::InferenceReport, name::Symbol) =
    name === :sections ? getfield(report, :sections) :
    getproperty(getfield(report, :sections), name)
Base.propertynames(report::InferenceReport) = keys(getfield(report, :sections))

report_tables(report::InferenceReport) = reduce(vcat, (report_tables(section) for section in getfield(report, :sections)); init = ReportTable[])

function Base.show(io::IO, ::MIME"text/plain", report::InferenceReport)
    print(io, "Inference report")
    for section in getfield(report, :sections)
        println(io)
        show(io, MIME"text/plain"(), section)
    end
    return nothing
end

"""
    RxInferReportCallbacks(report::InferenceReport)

The callbacks `infer(...; report = …)` records the `report` with: each event goes to the sections
that listen to it, and the report is saved in the model's metadata, `model.metadata[:report]`,
once the model is created.
"""
struct RxInferReportCallbacks{R <: InferenceReport}
    report::R
end

ReactiveMP.listens(callbacks::RxInferReportCallbacks, ::Type{T}) where {T} =
    T <: AfterModelCreationEvent || any(section -> ReactiveMP.listens(section, T), getfield(callbacks.report, :sections))

function ReactiveMP.handle_event(callbacks::RxInferReportCallbacks, event::AfterModelCreationEvent)
    haskey(event.model.metadata, :report) && error(
        "The model's metadata already contains a `:report` key. This can happen if you pass `report = …` while also providing `RxInferReportCallbacks` in the `callbacks` argument. Use one or the other, not both.",
    )
    event.model.metadata[:report] = callbacks.report
    record_sections!(callbacks, event)
    return nothing
end

ReactiveMP.handle_event(callbacks::RxInferReportCallbacks, event) = record_sections!(callbacks, event)

function record_sections!(callbacks::RxInferReportCallbacks, event)
    for section in getfield(callbacks.report, :sections)
        ReactiveMP.listens(section, typeof(event)) && record!(section, event)
    end
    return nothing
end
