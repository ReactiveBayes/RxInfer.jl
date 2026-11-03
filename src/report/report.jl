export InferenceReport, ApproximationsReport

import ReactiveMP:
    AfterMessageRuleCallEvent,
    AfterMarginalRuleCallEvent,
    AfterFactorBoundFreeEnergyEvent,
    AfterFormConstraintAppliedEvent,
    BeforeMarginalComputationEvent,
    AfterMarginalComputationEvent,
    WrappedFormConstraint,
    CompositeFormConstraint,
    UnspecifiedFormConstraint

"""
    InferenceReport

What `infer(...; report = true)` records about an inference, a section per subject, each read as a
property: `result.report.approximations` is the [`ApproximationsReport`](@ref). `report` names the
sections to record, `true` for all of them: `report = (:approximations,)`.

The report is collected from the engine's events, and costs nothing when off, the default. Read it
from the result of a batch inference, `result.report`, and from the engine of a streaming one,
`engine.model.metadata[:report]`.
"""
struct InferenceReport{S <: NamedTuple}
    sections::S
end

const REPORT_SECTIONS = (:approximations,)

new_report_section(::Val{:approximations}) = ApproximationsReport()

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

function Base.show(io::IO, ::MIME"text/plain", report::InferenceReport)
    print(io, "Inference report")
    for section in getfield(report, :sections)
        println(io)
        show(io, MIME"text/plain"(), section)
    end
    return nothing
end

"""
    RxInfer.NodeRecord

What an [`ApproximationsReport`](@ref) records of one kind of computation of a node: `rules`, how
many times an exact rule ran, and `rule`, the last; `fallbacks`, how many times the rule fallback
ran, and `fallback`, which; `backends`, the expectation backend each input's distribution type
went to under the fallback, as `(backend, exact)`.
"""
mutable struct NodeRecord
    rules::Int
    fallbacks::Int
    rule::Any
    fallback::Any
    backends::Dict{Any, Any}
end

NodeRecord() = NodeRecord(0, 0, nothing, nothing, Dict())

"""
    RxInfer.ProjectionRecord

How many times a form constraint projected a variable's marginal or messages, `projections`, and
the last projection's `outcome`, as `RxInfer.form_outcome` reports it.
"""
mutable struct ProjectionRecord
    projections::Int
    outcome::Any
end

ProjectionRecord() = ProjectionRecord(0, nothing)

"""
    RxInfer.VariableRecord

What an [`ApproximationsReport`](@ref) records of a random variable: its form `constraint`, and the
[`RxInfer.ProjectionRecord`](@ref)s of its `marginal` and its `messages`.
"""
mutable struct VariableRecord
    constraint::Any
    marginal::ProjectionRecord
    messages::ProjectionRecord
end

VariableRecord() = VariableRecord(nothing, ProjectionRecord(), ProjectionRecord())

"""
    ApproximationsReport

The section of an [`InferenceReport`](@ref) that says what was computed exactly and what was
approximated, collected from the engine's events:

- `nodes`, for each node type, what it computed and how: the key `(node, kind, target)`, `kind`
  being `:message`, `:marginal` or `:energy`, gives a [`RxInfer.NodeRecord`](@ref): how many
  times an exact rule ran and the rule fallback, the last rule (a `RuleSpec`, with its file and
  line), and the expectation backend each input went to under the fallback;
- `variables`, for each random variable by its name and index, a [`RxInfer.VariableRecord`](@ref):
  the form constraint that projected its beliefs, how many times, and the last projection's
  outcome, as the constraint reports it through [`RxInfer.form_outcome`](@ref) (for `LaplaceForm`,
  its `method`, `logZ`, `converged` and `iterations`), for the marginal and the messages apart;
- `expectation`, the expectation backend of the nodes' rule context, chosen once per inference.

This is the foundation of the report: the records are plain data, which a later presentation
summarises.
"""
mutable struct ApproximationsReport
    nodes::Dict{Tuple{Any, Symbol, Any}, NodeRecord}
    variables::Dict{Tuple{Symbol, Any}, VariableRecord}
    expectation::Any
    # the variables whose marginal is being computed, whose projections are the marginal's
    computing::Set{Any}
end

ApproximationsReport() = ApproximationsReport(Dict(), Dict(), nothing, Set())


# The node a rule event is about
event_node(event::AfterMessageRuleCallEvent) = ReactiveMP.message_mapping_fform(event.mapping)
event_node(event::AfterMarginalRuleCallEvent) = ReactiveMP.functionalform(event.mapping.factornode)
event_node(event::AfterFactorBoundFreeEnergyEvent) = ReactiveMP.functionalform(event.node)

event_key(event::AfterMessageRuleCallEvent) = (event_node(event), :message, report_target(event.mapping.target))
event_key(event::AfterMarginalRuleCallEvent) = (event_node(event), :marginal, report_target(event.mapping.target))

# A target as the report keys it: the interface's name, `(group, k)` for a member of a group, the
# members of a cluster
report_target(target::MessagePassingRulesBase.Target) = MessagePassingRulesBase.target_edge(target)
report_target(target::MessagePassingRulesBase.IndexedTarget) =
    (MessagePassingRulesBase.target_edge(target), MessagePassingRulesBase.target_index(target))
report_target(target::MessagePassingRulesBase.ClusterTarget) = MessagePassingRulesBase.cluster_members(target)
report_target(target) = target
event_key(event::AfterFactorBoundFreeEnergyEvent) = (event_node(event), :energy, nothing)

event_context(event::Union{AfterMessageRuleCallEvent, AfterMarginalRuleCallEvent}) = event.mapping.context
event_context(event::AfterFactorBoundFreeEnergyEvent) = nothing

# The distributions of a rule's inputs, whose expectations the fallback computed
event_inputs(event::Union{AfterMessageRuleCallEvent, AfterMarginalRuleCallEvent}) =
    (something_or_empty(event.messages)..., something_or_empty(event.marginals)...)
event_inputs(event::AfterFactorBoundFreeEnergyEvent) = something_or_empty(event.marginals)
something_or_empty(::Nothing) = ()
something_or_empty(inputs) = Tuple(inputs)

function record!(report::ApproximationsReport, event::Union{AfterMessageRuleCallEvent, AfterMarginalRuleCallEvent, AfterFactorBoundFreeEnergyEvent})
    rule = event.rule
    isnothing(rule) && return nothing
    record = get!(NodeRecord, report.nodes, event_key(event))
    context = event_context(event)
    if !isnothing(context) && isnothing(report.expectation)
        report.expectation = context.expectation
    end
    if rule isa MessagePassingRulesBase.RuleSpec
        record.rules += 1
        record.rule = rule
    else
        record.fallbacks += 1
        record.fallback = rule
        backend = isnothing(context) ? report.expectation : context.expectation
        isnothing(backend) || record_backends!(record.backends, backend, event_inputs(event))
    end
    return nothing
end

function record_backends!(backends, backend, inputs)
    for input in inputs
        distribution = ReactiveMP.getdata(input)
        Q = typeof(distribution)
        haskey(backends, Q) && continue
        chosen = BayesBase.which_expectation_backend(backend, Any, Q)
        exact = !isnothing(chosen) && BayesBase.has_expectation_label(backend, Any, Q, BayesBase.ExpectationExact())
        backends[Q] = (backend = chosen, exact = exact)
    end
    return backends
end

# A variable by its name and index, as the model names it
variable_key(label::GraphPPL.VariableNodeProperties) = (GraphPPL.getname(label), GraphPPL.index(label))
variable_key(label::Symbol) = (label, nothing)
variable_key(label) = (Symbol(repr(label)), nothing)

record!(report::ApproximationsReport, event::BeforeMarginalComputationEvent) =
    (push!(report.computing, event.variable); nothing)
record!(report::ApproximationsReport, event::AfterMarginalComputationEvent) =
    (delete!(report.computing, event.variable); nothing)

function record!(report::ApproximationsReport, event::AfterFormConstraintAppliedEvent)
    for part in form_parts(event.context.form_constraint)
        constraint, context = part
        is_reported_constraint(constraint) || continue
        record = get!(VariableRecord, report.variables, variable_key(event.variable.label))
        record.constraint = constraint
        projection = event.variable in report.computing ? record.marginal : record.messages
        projection.projections += 1
        projection.outcome = form_outcome(constraint, context, event.distribution, event.result)
    end
    return nothing
end

# The parts of a form constraint, each with its prepared context
form_parts(constraint::CompositeFormConstraint) =
    foldl((parts, c) -> (parts..., form_parts(c)...), constraint.constraints; init = ())
form_parts(constraint::WrappedFormConstraint) = ((constraint.constraint, constraint.context),)
form_parts(constraint) = ((constraint, nothing),)

# RxInfer's check of the form, and the absence of a constraint, are not reported
is_reported_constraint(::UnspecifiedFormConstraint) = false
is_reported_constraint(::EnsureSupportedFunctionalForm) = false
is_reported_constraint(constraint) = true

"""
    RxInfer.form_outcome(constraint, context, distribution, result)

What the form constraint `constraint`, with its prepared `context`, reports of its last projection,
of `distribution` onto `result`, to the [`ApproximationsReport`](@ref): by default whether it
changed the distribution, `(changed = …,)`.
"""
form_outcome(constraint, context, distribution, result) = (changed = result !== distribution,)

ReactiveMP.listens(::ApproximationsReport, ::Type{T}) where {T} =
    T <: Union{
    AfterMessageRuleCallEvent, AfterMarginalRuleCallEvent, AfterFactorBoundFreeEnergyEvent,
    AfterFormConstraintAppliedEvent, BeforeMarginalComputationEvent, AfterMarginalComputationEvent,
}

function Base.show(io::IO, ::MIME"text/plain", report::ApproximationsReport)
    println(io, "  Approximations")
    println(io, "    Nodes (exact rules / fallback)")
    for key in sort!(collect(keys(report.nodes)); by = repr)
        (node, kind, target) = key
        record = report.nodes[key]
        what = isnothing(target) ? string(kind) : string(kind, " towards ", target_display(target))
        print(io, "      ", node_name(node), ", ", what, ": ", record.rules, " / ", record.fallbacks)
        if !isempty(record.backends)
            print(io, ", expectations by ")
            join(io, (string(nameof(Q), " → ", backend_name(entry)) for (Q, entry) in record.backends), ", ")
        end
        println(io)
    end
    println(io, "    Variables")
    for key in sort!(collect(keys(report.variables)); by = repr)
        record = report.variables[key]
        print(io, "      ", variable_display(key), ": ", nameof(typeof(record.constraint)))
        for (name, projection) in (("marginal", record.marginal), ("messages", record.messages))
            projection.projections > 0 || continue
            print(io, "; ", name, " ", outcome_display(projection.outcome), " (", projection.projections, ")")
        end
        println(io)
    end
    print(io, "    Expectation backend: ", expectation_display(report.expectation))
    return nothing
end

node_name(node) = string(ReactiveMP.node_display(node))
target_display(target) = repr(target)
backend_name(entry) = isnothing(entry.backend) ? "none" : string(nameof(typeof(entry.backend)), entry.exact ? " (exact)" : "")
variable_display((name, index)) = isnothing(index) ? string(name) : string(name, "[", join(index, ", "), "]")
outcome_display(::Nothing) = "projected"
expectation_display(::Nothing) = "none"
expectation_display(backend::BayesBase.ExpectationFirstOf) =
    string("ExpectationFirstOf(", join((expectation_display(b) for b in backend.backends), ", "), ")")
expectation_display(backend) = string(nameof(typeof(backend)))
outcome_display(outcome::NamedTuple) = join((string(k, " = ", v) for (k, v) in pairs(outcome)), ", ")

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
    return nothing
end

function ReactiveMP.handle_event(callbacks::RxInferReportCallbacks, event)
    for section in getfield(callbacks.report, :sections)
        ReactiveMP.listens(section, typeof(event)) && record!(section, event)
    end
    return nothing
end
