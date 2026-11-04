export ApproximationsReport

import ReactiveMP:
    AfterMessageRuleCallEvent,
    AfterMarginalRuleCallEvent,
    AfterFactorBoundFreeEnergyEvent,
    AfterFormConstraintAppliedEvent,
    AfterSiteComputationEvent,
    BeforeMarginalComputationEvent,
    AfterMarginalComputationEvent,
    WrappedFormConstraint,
    CompositeFormConstraint,
    UnspecifiedFormConstraint

"""
    RxInfer.NodeRecord

What an [`ApproximationsReport`](@ref) records of one kind of computation of a node, of its type or
of one [`RxInfer.NodeInstance`](@ref):
- `rules`, how many times an exact rule ran, and `rule`, the last;
- `fallbacks`, how many times the rule fallback ran, and `fallback`, which;
- `backends`, the expectation backend each input's distribution type went to under the fallback,
  as `(backend, exact, stochastic, priority)`;
- `outcome`, how the fallback computed its last answer, as it declared it (such as a joint
  Laplace approximation's `(method, converged, iterations)`), and `nonconverged`, how many of its
  answers did not converge;
- `over_messages`, how many of the fallback's messages took an expectation over another random
  variable's message, as under belief propagation, where the expected log-density approximates
  the message.
"""
mutable struct NodeRecord
    rules::Int
    fallbacks::Int
    rule::Any
    fallback::Any
    backends::Dict{Any, Any}
    outcome::Any
    nonconverged::Int
    over_messages::Int
end

NodeRecord() = NodeRecord(0, 0, nothing, nothing, Dict(), nothing, 0, 0)

"""
    RxInfer.NodeInstance

One factor node of the model in an [`ApproximationsReport`](@ref): its type `node`, how the model
writes it, its `label`, such as `y[3] ~ Bernoulli(p[3])` or `p[3] := logistic(·)`, and what it
computed, `records`, a [`RxInfer.NodeRecord`](@ref) for each `(kind, target)`.
"""
struct NodeInstance
    node::Any
    head::String
    out::Tuple{String, Any}
    inputs::Vector{Tuple{String, Any}}
    records::Dict{Tuple{Symbol, Any}, NodeRecord}
end

function NodeInstance(factornode)
    node = ReactiveMP.functionalform(factornode)
    interfaces = ReactiveMP.getinterfaces(factornode)
    refs = [(ReactiveMP.name(interface), variable_ref(ReactiveMP.getvariable(interface))) for interface in interfaces]
    out = something(findfirst(((name, _),) -> name === :out, refs), 1)
    if node === StandaloneDistribution
        # `x ~ d` for a distribution value: the node is the distribution's type
        distribution = ReactiveMP.getvariable(interfaces[2]).constant
        return NodeInstance(node, string("~ ", nameof(typeof(distribution))), last(refs[out]), Tuple{String, Any}[], Dict())
    end
    deterministic = MessagePassingRulesBase.sdtype(node) isa MessagePassingRulesBase.Deterministic
    head = string(deterministic ? ":= " : "~ ", node_function_name(node))
    inputs = Tuple{String, Any}[last(ref) for (k, ref) in enumerate(refs) if k != out]
    return NodeInstance(node, head, last(refs[out]), inputs, Dict())
end

Base.getproperty(instance::NodeInstance, name::Symbol) =
    name === :label ? instances_label([instance]) : getfield(instance, name)

# A Delta node by its function, any other by its type: `Categorical` by its own name, not the
# `DiscreteNonParametric` it is an alias of, and `x ~ d` for a distribution value as a distribution
node_function_name(::Type{DeltaFn{F}}) where {F} = isdefined(F, :instance) ? string(nameof(F.instance)) : string(F)
node_function_name(node) =
    node === Distributions.Categorical ? "Categorical" : node === StandaloneDistribution ? "distribution" : string(ReactiveMP.node_display(node))

# A variable as a label shows it: its name and index, an anonymous one as `·`, a constant by its
# value, which instances of one shape may differ in
const CONSTANT_REF = "\0constant"

variable_ref(variable) = label_ref(variable.label, variable)
function label_ref(label::GraphPPL.VariableNodeProperties, variable)
    GraphPPL.is_anonymous(label) && return ("·", nothing)
    GraphPPL.is_constant(label) && return (CONSTANT_REF, constant_display(GraphPPL.value(label)))
    return (string(GraphPPL.getname(label)), GraphPPL.index(label))
end
label_ref(label::Symbol, variable) = (string(label), nothing)
label_ref(::Nothing, variable::ReactiveMP.ConstVariable) = (CONSTANT_REF, constant_display(variable.constant))
label_ref(label, variable) = ("·", nothing)

constant_display(value::Real) = string(round(value; sigdigits = 4))
constant_display(value::AbstractVector{<:Real}) = length(value) <= 3 ? string("[", join(map(constant_display, value), ", "), "]") : "const"
constant_display(value) = "const"

# The instances' shape: a node type with its variables by name, under which instances fold
instance_shape(instance::NodeInstance) = (instance.node, instance.head, first(instance.out), map(first, instance.inputs))

# Instances of one shape as one label, their indices folded: `y[1:4] ~ BinomialLogistic(θ, ·)`
function instances_label(instances)
    instance = first(instances)
    out = ref_display(map(i -> i.out, instances))
    isempty(instance.inputs) && return string(out, " ", instance.head)
    inputs = (ref_display(map(i -> i.inputs[k], instances)) for k in eachindex(instance.inputs))
    return string(out, " ", instance.head, "(", join(inputs, ", "), ")")
end

function ref_display(refs)
    name = first(first(refs))
    # constants by their value, where the instances agree on it
    name == CONSTANT_REF && return (values = unique(last.(refs)); length(values) == 1 ? only(values) : "…")
    indices = [last(ref) for ref in refs]
    all(isnothing, indices) && return name
    distinct = unique(indices)
    length(distinct) == 1 && return string(name, "[", single_index_display(only(distinct)), "]")
    return string(name, "[", index_display(filter(!isnothing, indices)), "]")
end

single_index_display(index::Tuple) = join(index, ", ")
single_index_display(index) = string(index)

"""
    RxInfer.ProjectionRecord

How many times a form constraint projected a variable's marginal or messages, `projections`, and
the last projection's `outcome`, as `RxInfer.form_outcome` reports it; of them, `nonconverged`
did not converge and `unprojected` were left as they were (`LaplaceForm(nomode = :leave)`).
"""
mutable struct ProjectionRecord
    projections::Int
    outcome::Any
    nonconverged::Int
    unprojected::Int
end

ProjectionRecord() = ProjectionRecord(0, nothing, 0, 0)

"""
    RxInfer.VariableRecord

What an [`ApproximationsReport`](@ref) records of a random variable: its form `constraint`, the
[`RxInfer.ProjectionRecord`](@ref)s of its `marginal` and its `messages`, and how its posterior was
formed, `incoming`: for each node and target whose messages reached the variable, how many came
from an exact rule and how many from the rule fallback, `(rules, fallbacks)`. Under cavity
propagation, `sites` records the sites projected on its edges.
"""
mutable struct VariableRecord
    constraint::Any
    marginal::ProjectionRecord
    messages::ProjectionRecord
    incoming::Dict{Tuple{Any, Any}, Tuple{Int, Int}}
    sites::ProjectionRecord
end

VariableRecord() = VariableRecord(nothing, ProjectionRecord(), ProjectionRecord(), Dict(), ProjectionRecord())

"""
    ApproximationsReport

The section of an [`InferenceReport`](@ref) that says what was computed exactly and what was
approximated, collected from the engine's events. Shown by posterior: each approximated posterior
with how it was projected and the nodes whose messages formed it, exact or by the rule fallback,
then the joints and energies the fallback computed, after the warnings of
[`RxInfer.diagnostics`](@ref). Its records are plain data:

- `instances`, each [`RxInfer.NodeInstance`](@ref) that computed something, by its factor node:
  its label in the model and a [`RxInfer.NodeRecord`](@ref) for each `(kind, target)`, `kind`
  being `:message`, `:marginal` or `:energy`;
- `nodes`, the same by node type, keyed by `(node, kind, target)`;
- `variables`, for each random variable by its name and index, a [`RxInfer.VariableRecord`](@ref):
  the form constraint that projected its beliefs, how, and the messages that formed its posterior;
- `expectation`, the expectation backend of the nodes' rule context, chosen once per inference.

[`report_tables`](@ref) gives the tables it is shown as.
"""
mutable struct ApproximationsReport
    nodes::Dict{Tuple{Any, Symbol, Any}, NodeRecord}
    instances::IdDict{Any, NodeInstance}
    variables::Dict{Tuple{Symbol, Any}, VariableRecord}
    expectation::Any
    # the variables whose marginal is being computed, whose projections are the marginal's
    computing::Set{Any}
end

ApproximationsReport() = ApproximationsReport(Dict(), IdDict(), Dict(), nothing, Set())

# The node a rule event is about, its type and its factor node
event_node(event::AfterMessageRuleCallEvent) = ReactiveMP.message_mapping_fform(event.mapping)
event_node(event::AfterMarginalRuleCallEvent) = ReactiveMP.functionalform(event.mapping.factornode)
event_node(event::AfterFactorBoundFreeEnergyEvent) = ReactiveMP.functionalform(event.node)

event_factornode(event::Union{AfterMessageRuleCallEvent, AfterMarginalRuleCallEvent}) = event.mapping.factornode
event_factornode(event::AfterFactorBoundFreeEnergyEvent) = event.node

event_kind(::AfterMessageRuleCallEvent) = :message
event_kind(::AfterMarginalRuleCallEvent) = :marginal
event_kind(::AfterFactorBoundFreeEnergyEvent) = :energy

event_target(event::Union{AfterMessageRuleCallEvent, AfterMarginalRuleCallEvent}) = report_target(event.mapping.target)
event_target(event::AfterFactorBoundFreeEnergyEvent) = nothing

# A target as the report keys it: the interface's name, `(group, k)` for a member of a group, the
# members of a cluster
report_target(target::MessagePassingRulesBase.Target) = MessagePassingRulesBase.target_edge(target)
report_target(target::MessagePassingRulesBase.IndexedTarget) =
    (MessagePassingRulesBase.target_edge(target), MessagePassingRulesBase.target_index(target))
report_target(target::MessagePassingRulesBase.ClusterTarget) = MessagePassingRulesBase.cluster_members(target)
report_target(target) = target

event_context(event::Union{AfterMessageRuleCallEvent, AfterMarginalRuleCallEvent}) = event.mapping.context
event_context(event::AfterFactorBoundFreeEnergyEvent) = nothing

# The distributions of a rule's inputs, whose expectations the fallback computed: a
# deterministic node's message on `out` is evaluated, not integrated, so it is left out
event_inputs(event::Union{AfterMessageRuleCallEvent, AfterMarginalRuleCallEvent}) =
    (expected_messages(event)..., something_or_empty(event.marginals)...)
event_inputs(event::AfterFactorBoundFreeEnergyEvent) = something_or_empty(event.marginals)

function expected_messages(event)
    messages = something_or_empty(event.messages)
    isempty(messages) && return messages
    MessagePassingRulesBase.sdtype(event_node(event)) isa MessagePassingRulesBase.Deterministic || return messages
    names = ReactiveMP.unval(event.mapping.msgs_names)
    return Tuple(message for (name, message) in zip(names, messages) if name !== :out)
end
something_or_empty(::Nothing) = ()
something_or_empty(inputs) = Tuple(inputs)

function record!(report::ApproximationsReport, event::Union{AfterMessageRuleCallEvent, AfterMarginalRuleCallEvent, AfterFactorBoundFreeEnergyEvent})
    rule = event.rule
    isnothing(rule) && return nothing
    node, kind, target = event_node(event), event_kind(event), event_target(event)
    context = event_context(event)
    if !isnothing(context) && isnothing(report.expectation)
        report.expectation = context.expectation
    end
    factornode = event_factornode(event)
    instance = get!(() -> NodeInstance(factornode), report.instances, factornode)
    backend = isnothing(context) ? report.expectation : context.expectation
    over_messages = !(rule isa MessagePassingRulesBase.RuleSpec) && event isa AfterMessageRuleCallEvent && expects_over_messages(event, node)
    for record in (get!(NodeRecord, report.nodes, (node, kind, target)), get!(NodeRecord, instance.records, (kind, target)))
        count_call!(record, rule, event, backend, over_messages)
    end
    event isa AfterMessageRuleCallEvent && record_incoming!(report, factornode, event.mapping.target, target, rule)
    return nothing
end

function count_call!(record::NodeRecord, rule, event, backend, over_messages)
    if rule isa MessagePassingRulesBase.RuleSpec
        record.rules += 1
        record.rule = rule
        return record
    end
    record.fallbacks += 1
    record.fallback = rule
    record.over_messages += over_messages
    outcome = event.outcome
    if !isnothing(outcome)
        record.outcome = outcome
        outcome isa NamedTuple && get(outcome, :converged, true) === false && (record.nonconverged += 1)
    end
    isnothing(backend) || record_backends!(record.backends, backend, event_inputs(event))
    return record
end

function record_backends!(backends, backend, inputs)
    for input in inputs
        foreach_distribution(input) do distribution
            Q = typeof(distribution)
            haskey(backends, Q) && return nothing
            chosen = BayesBase.which_expectation_backend(backend, Any, Q)
            exact = !isnothing(chosen) && BayesBase.has_expectation_label(backend, Any, Q, BayesBase.ExpectationExact())
            stochastic = !isnothing(chosen) && BayesBase.has_expectation_label(backend, Any, Q, BayesBase.ExpectationStochastic())
            priority = isnothing(chosen) ? nothing : BayesBase.expectation_backend_priority(chosen)
            backends[Q] = (backend = chosen, exact = exact, stochastic = stochastic, priority = priority)
            return nothing
        end
    end
    return backends
end

# An input of a rule is a message or a marginal, or a tuple of them for the members of a group
foreach_distribution(f, input::Tuple) = foreach(i -> foreach_distribution(f, i), input)
foreach_distribution(f, input) = f(ReactiveMP.getdata(input))

# Whether a fallback's message took an expectation over another random variable's message, as
# under belief propagation: the expected log-density then approximates the message. A
# deterministic node's message towards `out` is its pushforward, and towards an input the message
# on `out` is what is pulled back, so only its other inputs count.
function expects_over_messages(event::AfterMessageRuleCallEvent, node)
    isnothing(event.messages) && return false
    deterministic = MessagePassingRulesBase.sdtype(node) isa MessagePassingRulesBase.Deterministic
    deterministic && event_target(event) === :out && return false
    names = ReactiveMP.unval(event.mapping.msgs_names)
    for (name, input) in zip(names, event.messages)
        deterministic && name === :out && continue
        spread = false
        foreach_distribution(d -> (spread |= !(d isa PointMass)), input)
        spread && return true
    end
    return false
end

# The posterior's provenance: a message towards a random variable, from an exact rule or the
# fallback
function record_incoming!(report::ApproximationsReport, factornode, target, key, rule)
    variable = target_variable(factornode, target)
    (isnothing(variable) || !ReactiveMP.israndom(variable)) && return nothing
    record = get!(VariableRecord, report.variables, variable_key(variable.label))
    rules, fallbacks = get(record.incoming, (factornode, key), (0, 0))
    exact = rule isa MessagePassingRulesBase.RuleSpec
    record.incoming[(factornode, key)] = (rules + exact, fallbacks + !exact)
    return nothing
end

function target_variable(factornode, target::MessagePassingRulesBase.Target)
    edge = MessagePassingRulesBase.target_edge(target)
    for interface in ReactiveMP.getinterfaces(factornode)
        interface isa ReactiveMP.NodeInterface && ReactiveMP.name(interface) === edge && return ReactiveMP.getvariable(interface)
    end
    return nothing
end

function target_variable(factornode, target::MessagePassingRulesBase.IndexedTarget)
    edge, k = MessagePassingRulesBase.target_edge(target), MessagePassingRulesBase.target_index(target)
    for interface in ReactiveMP.getinterfaces(factornode)
        interface isa ReactiveMP.IndexedNodeInterface && ReactiveMP.name(interface) === edge && interface.index == k &&
            return ReactiveMP.getvariable(interface)
    end
    return nothing
end

target_variable(factornode, target) = nothing

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
        outcome = form_outcome(constraint, context, event.distribution, event.result)
        projection.outcome = outcome
        if outcome isa NamedTuple
            unprojected = get(outcome, :method, nothing) === :unprojected
            projection.unprojected += unprojected
            projection.nonconverged += !unprojected && get(outcome, :converged, true) === false
        end
    end
    return nothing
end

# A site projected on one of a variable's edges, under cavity propagation; a message that is its
# own site, or passes, is not a projection
function record!(report::ApproximationsReport, event::AfterSiteComputationEvent)
    outcome = event.outcome
    (outcome isa NamedTuple && get(outcome, :method, nothing) in (:laplace, :moment_matching)) || return nothing
    sites = get!(VariableRecord, report.variables, variable_key(event.variable.label)).sites
    sites.projections += 1
    sites.outcome = outcome
    sites.nonconverged += get(outcome, :converged, true) === false
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
    AfterSiteComputationEvent,
}

"""
    RxInfer.node_types(report::ApproximationsReport)

The records by node type, `(node, kind, target) => RxInfer.NodeRecord`, every instance of a type
counted together.
"""
node_types(report::ApproximationsReport) = report.nodes

# Whether a posterior was approximated: a message from the fallback formed it, or its projection
# changed it beyond normalising a closed form
function is_approximated(record::VariableRecord)
    any(((rules, fallbacks),) -> fallbacks > 0, values(record.incoming)) && return true
    record.sites.projections > 0 && return true
    return record.marginal.projections > 0 && projection_approximates(record.marginal.outcome)
end

projection_approximates(outcome::NamedTuple) =
    haskey(outcome, :method) ? outcome.method ∉ (:closed_form, :point_mass, :discrete) : get(outcome, :changed, true)
projection_approximates(outcome) = true

# The approximated variables, folded by name where their posteriors were formed alike
function approximated_groups(report::ApproximationsReport)
    groups = Dict{Any, Vector{Tuple{Symbol, Any}}}()
    order = Any[]
    for key in sort!(collect(keys(report.variables)); by = variable_order)
        record = report.variables[key]
        is_approximated(record) || continue
        signature = (first(key), variable_signature(report, record))
        haskey(groups, signature) || push!(order, signature)
        push!(get!(groups, signature, Tuple{Symbol, Any}[]), key)
    end
    return [groups[signature] for signature in order]
end

variable_order((name, index)) = (string(name), isnothing(index) ? () : index isa Tuple ? index : (index,))

function variable_signature(report::ApproximationsReport, record::VariableRecord)
    constraint = isnothing(record.constraint) ? nothing : nameof(typeof(record.constraint))
    method = record.marginal.outcome isa NamedTuple ? get(record.marginal.outcome, :method, nothing) : nothing
    incoming = sort!([(repr(instance_shape(report.instances[factornode])), repr(target), rules > 0, fallbacks > 0) for ((factornode, target), (rules, fallbacks)) in record.incoming])
    return (constraint, method, Tuple(incoming))
end

# Variables as a label shows them, their indices folded: `x[1:200]`; the model's anonymous ones,
# such as the output of `dot` inside `logistic(dot(a, θ))`, as `·`
variables_display(keys) =
    first(first(keys)) === GraphPPL.VariableNameAnonymous ? "· (anonymous)" : ref_display([(string(name), index) for (name, index) in keys])

function projection_display(records)
    record = first(records)
    any(r -> r.sites.projections > 0, records) && return sites_display(records)
    projected = [r.marginal for r in records if r.marginal.projections > 0]
    if isempty(projected)
        isnothing(record.constraint) && return "no projection"
        return string(nameof(typeof(record.constraint)), " on its messages only")
    end
    outcomes = [p.outcome for p in projected]
    outcome = first(outcomes)
    outcome isa NamedTuple && haskey(outcome, :method) || return string(nameof(typeof(record.constraint)), ", ", outcome_display(outcome))
    method = outcome.method
    if method === :projection
        text = string("projected onto ", get(outcome, :family, "a family"))
        bound = get(outcome, :logZ, nothing)
        isnothing(bound) || (text *= string(", ELBO = ", round(bound; sigdigits = 4)))
        return text
    end
    method === :closed_form && return "closed form"
    method === :discrete && return "normalised exactly"
    method === :point_mass && return "a point mass"
    method === :unprojected && return "left unprojected"
    converged = count(o -> get(o, :converged, true) === true, outcomes)
    iterations = [o.iterations for o in outcomes if haskey(o, :iterations)]
    name = method === :laplace ? "Laplace" : method === :variational ? "CVI" : string(method)
    bound = method === :variational ? "ELBO" : "log Z"
    if length(outcomes) == 1
        steps = string(only(iterations), only(iterations) == 1 ? " iteration" : " iterations")
        text = converged == 1 ? string(name, ", converged in ", steps) : string(name, ", did not converge in ", steps)
        haskey(outcome, :logZ) && !isnothing(outcome.logZ) && (text *= string(", ", bound, " = ", round(outcome.logZ; sigdigits = 4)))
        return text
    end
    span = minimum(iterations) == maximum(iterations) ? string(minimum(iterations)) : string(minimum(iterations), " to ", maximum(iterations))
    return string(name, ", ", converged, " of ", length(outcomes), " converged, ", span, " iterations")
end

# Under cavity propagation: the sites, by the projection of the scheme, and how they converged
function sites_display(records)
    sites = [r.sites for r in records if r.sites.projections > 0]
    method = get(first(sites).outcome, :method, nothing)
    name = method === :laplace ? "Laplace propagation" : "expectation propagation"
    total = sum(s -> s.projections, sites)
    nonconverged = sum(s -> s.nonconverged, sites)
    text = string(name, ", ", total, total == 1 ? " site" : " sites")
    nonconverged > 0 && (text *= string(", ", nonconverged, " not converged"))
    return text
end

outcome_display(::Nothing) = "projected"
outcome_display(outcome::NamedTuple) = join((string(k, " = ", v) for (k, v) in pairs(outcome)), ", ")
outcome_display(outcome) = string(outcome)

target_display(target::Symbol) = string(target)
target_display((group, k)::Tuple{Symbol, Integer}) = string(group, "[", k, "]")
target_display(target::Tuple) = string("(", join(target, ", "), ")")
target_display(target) = repr(target)

# The expectations of the fallback's inputs, in a few words: `exact`, or the backends that are not
function expectations_display(backends)
    isempty(backends) && return ""
    entries = collect(backends)
    all(((_, entry),) -> entry.exact, entries) && return "exact"
    inexact = [string(nameof(Q), " → ", backend_name(entry)) for (Q, entry) in entries if !entry.exact]
    return join(sort!(unique(inexact)), ", ")
end

backend_name(entry) = isnothing(entry.backend) ? "none" : string(nameof(typeof(entry.backend)), entry.exact ? " (exact)" : "")

function report_tables(report::ApproximationsReport)
    tables = ReportTable[]
    found = diagnostics(report)
    isempty(found) || push!(tables, diagnostics_table(found))
    push!(tables, posteriors_table(report))
    push!(tables, fallback_table(report))
    return filter(!isempty, tables)
end

# The approximated posteriors, each a group of rows: the nodes whose messages formed it
function posteriors_table(report::ApproximationsReport)
    rows, groups = Vector{String}[], Pair{Int, String}[]
    for keys in approximated_groups(report)
        records = [report.variables[key] for key in keys]
        push!(groups, length(rows) + 1 => string(variables_display(keys), ": ", projection_display(records)))
        incoming = Dict{Any, Any}()
        for record in records, ((factornode, target), (rules, fallbacks)) in record.incoming
            instance = report.instances[factornode]
            entry = get!(() -> (instances = NodeInstance[], target = target, rules = Ref(0), fallbacks = Ref(0), backends = Dict{Any, Any}()), incoming, (instance_shape(instance), target))
            push!(entry.instances, instance)
            entry.rules[] += rules
            entry.fallbacks[] += fallbacks
            merge!(entry.backends, get(instance.records, (:message, target), NodeRecord()).backends)
        end
        for entry in sort!(collect(values(incoming)); by = e -> (e.fallbacks[] > 0, instances_label(e.instances)))
            push!(rows, [instances_label(unique(entry.instances)), target_display(entry.target), string(entry.rules[]), string(entry.fallbacks[]), expectations_display(entry.backends)])
        end
        isempty(first(records).incoming) && push!(rows, ["(no messages recorded)", "", "", "", ""])
    end
    return ReportTable("Approximated posteriors", ["Messages from", "towards", "exact", "fallback", "expectations"], rows; groups)
end

# The joints and the energies the fallback computed, by node
function fallback_table(report::ApproximationsReport)
    entries = Dict{Any, Any}()
    for instance in values(report.instances), ((kind, target), record) in instance.records
        kind === :message && continue
        record.fallbacks > 0 || continue
        entry = get!(() -> (instances = NodeInstance[], kind = kind, target = target, fallbacks = Ref(0), nonconverged = Ref(0), outcome = Ref{Any}(nothing)), entries, (instance_shape(instance), kind, target))
        push!(entry.instances, instance)
        entry.fallbacks[] += record.fallbacks
        entry.nonconverged[] += record.nonconverged
        isnothing(record.outcome) || (entry.outcome[] = record.outcome)
    end
    rows, levels = Vector{String}[], Symbol[]
    for entry in sort!(collect(values(entries)); by = e -> (instances_label(e.instances), string(e.kind)))
        what = entry.kind === :energy ? "average energy" : string("joint of ", target_display(entry.target))
        how = fallback_outcome_display(entry.outcome[], entry.fallbacks[], entry.nonconverged[])
        push!(rows, [instances_label(entry.instances), what, string(entry.fallbacks[]), how])
        push!(levels, entry.nonconverged[] > 0 ? :warning : :none)
    end
    return ReportTable("Joints and energies by the fallback", ["Node", "computed", "times", "how"], rows; levels)
end

function fallback_outcome_display(outcome, calls, nonconverged)
    outcome isa NamedTuple && haskey(outcome, :method) || return "expectation"
    # a member's message times the node's own, with the other members known
    outcome.method === :closed_form && return "closed form"
    name = outcome.method === :joint_laplace ? "joint Laplace" : string(outcome.method)
    nonconverged == 0 && return string(name, ", converged")
    return string(name, ", ", nonconverged, " of ", calls, " did not converge")
end

function report_summary(report::ApproximationsReport)
    approximated = sum(length, approximated_groups(report); init = 0)
    total = length(report.variables)
    exact = [key for (key, record) in report.variables if !is_approximated(record)]
    found = diagnostics(report)
    warnings = count(d -> d.level === :warning, found)
    notes = length(found) - warnings
    text = string("Approximations: ", approximated, " of ", total, " posteriors approximated")
    if !isempty(exact)
        names = unique(first.(exact))
        shown = [variables_display(filter(k -> first(k) === name, exact)) for name in sort!(names; by = string)]
        text *= string(" (exact: ", join(shown[1:min(end, 5)], ", "), length(shown) > 5 ? ", …" : "", ")")
    end
    text *= warnings == 0 ? ", no warnings" : string(", ", warnings, warnings == 1 ? " warning" : " warnings")
    notes > 0 && (text *= string(", ", notes, notes == 1 ? " note" : " notes"))
    return text
end

report_footer(report::ApproximationsReport) = string("Expectations: ", expectation_display(report.expectation))

Base.show(io::IO, ::MIME"text/plain", report::ApproximationsReport) = show_report_section(io, report)

expectation_display(::Nothing) = "none"
expectation_display(backend::BayesBase.ExpectationFirstOf) =
    join((expectation_display(b) for b in backend.backends), ", ")
expectation_display(backend) = string(nameof(typeof(backend)))
