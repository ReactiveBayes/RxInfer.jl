const CompiledMarginalKey = GraphPPL.NodeDataExtraKey{
    :compiled_marginal, Int32
}()
const CompiledObservationKey = GraphPPL.NodeDataExtraKey{
    :compiled_observation, Int32
}()
const CompiledPredictionKey = GraphPPL.NodeDataExtraKey{
    :compiled_prediction, Int32
}()

struct CompiledOptionsPlugin end
GraphPPL.plugin_type(::CompiledOptionsPlugin) = GraphPPL.FactorNodePlugin()
function GraphPPL.preprocess_plugin(
    ::CompiledOptionsPlugin,
    model::GraphPPL.Model,
    context::GraphPPL.Context,
    label::GraphPPL.NodeLabel,
    node::GraphPPL.NodeData,
    options::GraphPPL.NodeCreationOptions,
)
    for key in
        (ReactiveMPExtraDependenciesKey, ReactiveMPExtraStreamPostprocessorsKey)
        name = GraphPPL.getkey(key)
        haskey(options, name) && setextra!(node, key, options[name])
    end
    return label, node
end
GraphPPL.postprocess_plugin(::CompiledOptionsPlugin, model::GraphPPL.Model) =
    nothing

# These callbacks run only at the serial API boundary; they are never invoked
# by message/product kernels. All other callbacks remain conservatively serial.
const CompiledBoundaryEvents = (
    :before_model_creation,
    :after_model_creation,
    :before_inference,
    :after_inference,
    :before_iteration,
    :after_iteration,
    :before_data_update,
    :after_data_update,
    :on_marginal_update,
    :before_autostart,
    :after_autostart,
)
ReactiveMP.compiled_callbacks_parallel_safe(
    callbacks::NamedTuple{K}
) where {K} = all(key -> key in CompiledBoundaryEvents, K)

function compiled_numerical_callbacks(options)
    callbacks = getcallbacks(options)
    if callbacks isa NamedTuple
        numerical = (;
            (
                key => value for
                (key, value) in pairs(callbacks) if key ∉ CompiledBoundaryEvents
            )...
        )
        return isempty(numerical) ? nothing : numerical
    end
    return callbacks
end

compiled_backend_requested(options) =
    options isa NamedTuple && get(options, :runner, nothing) isa CompiledRunner

"Flat lowering worklists, sharing node offsets/counts between both directions."
struct CompiledEdgeBindings
    offsets::Vector{Int32}
    counts::Vector{Int32}
    slots::Vector{Int32}
end
CompiledEdgeBindings() = CompiledEdgeBindings(Int32[], Int32[], Int32[])
function Base.getindex(edges::CompiledEdgeBindings, node::Int)
    count = edges.counts[node]
    iszero(count) && return nothing
    first = edges.offsets[node]
    return view(edges.slots, first:(first + count - 1))
end
function compiled_edge_bindings(graph)
    n = GraphPPL.nv(graph)
    offsets, counts = Vector{Int32}(undef, n + 1), zeros(Int32, n)
    next = 1
    for label in GraphPPL.labels(graph)
        offsets[label.global_counter] = next
        properties = getproperties(graph[label])
        if GraphPPL.is_variable(properties) && !GraphPPL.is_constant(properties)
            next += GraphPPL.degree(graph, label)
            next <= typemax(Int32) ||
                throw(OverflowError("Compiled edge-binding capacity exceeded"))
        end
    end
    offsets[end] = next
    return CompiledEdgeBindings(
        offsets, counts, Vector{Int32}(undef, next - 1)
    ),
    CompiledEdgeBindings(offsets, counts, Vector{Int32}(undef, next - 1))
end

mutable struct CompiledModelBuilder
    model::ProbabilisticModel
    program::ReactiveMP.CompiledProgram
    options::ReactiveMPInferenceOptions
    incoming::CompiledEdgeBindings
    outgoing::CompiledEdgeBindings
    score_slots::Vector{Int32}
    score_type::Any
    point_entropy_count::Int
end

function compiled_context(options, nodedata, which::Symbol)
    marginal = which === :marginal
    prefix = marginal ? :marginal : :messages
    formkey = if marginal
        GraphPPL.VariationalConstraintsMarginalFormConstraintKey
    else
        GraphPPL.VariationalConstraintsMessagesFormConstraintKey
    end
    form = ReactiveMP.preprocess_form_constraints(
        getextra(nodedata, formkey, ReactiveMP.UnspecifiedFormConstraint())
    )
    properties = getproperties(nodedata)
    form += EnsureSupportedFunctionalForm(
        marginal ? :q : :μ,
        GraphPPL.getname(properties),
        GraphPPL.index(properties),
    )
    return ReactiveMP.MessageProductContext(;
        fold_strategy = getextra(
            nodedata,
            Symbol(prefix, :_fold_strategy),
            ReactiveMP.MessagesProductFromLeftToRight(),
        ),
        prod_constraint = getextra(
            nodedata,
            Symbol(prefix, :_prod_constraint),
            ReactiveMP.default_prod_constraint(form),
        ),
        form_constraint = form,
        form_constraint_check_strategy = getextra(
            nodedata,
            Symbol(prefix, :_form_constraint_check_strategy),
            ReactiveMP.default_form_check_strategy(form),
        ),
        callbacks = compiled_numerical_callbacks(options),
        annotations = getannotations(options),
    )
end

function compiled_prepare_variables!(builder::CompiledModelBuilder)
    graph, program = getmodel(builder.model), builder.program
    variable_nodes(graph) do label, node
        properties = getproperties(node)
        kind = if is_random(properties)
            UInt8(0)
        elseif is_data(properties)
            UInt8(1)
        else
            UInt8(2)
        end
        d = GraphPPL.degree(graph, label)
        kind == 0 &&
            d < 2 &&
            throw(
                ArgumentError(
                    "Random variable $label must have at least two connected factors",
                ),
            )
        variable = ReactiveMP.CompiledVariable(
            Int32(label.global_counter), properties, Int32(d), kind
        )
        setextra!(node, ReactiveMPExtraVariableKey, variable)
        initial = getextra(node, InitMarExtraKey, nothing)
        initial = initial === nothing ? nothing : Marginal(initial, false, true)
        marginal = ReactiveMP.compiled_slot!(program, initial)
        setextra!(node, CompiledMarginalKey, marginal)
        if kind != 0
            value = if kind == 2
                Message(PointMass(GraphPPL.value(properties)), true, false)
            else
                nothing
            end
            observation = ReactiveMP.compiled_slot!(program, value)
            setextra!(node, CompiledObservationKey, observation)
            ReactiveMP.compiled_operation!(
                program,
                ReactiveMP.CompiledAsMarginalKernel(),
                marginal,
                observation,
            )
            builder.point_entropy_count += d
        end
        if kind == 1
            setextra!(
                node, CompiledPredictionKey, ReactiveMP.compiled_slot!(program)
            )
        end
    end
    return builder
end

function compiled_append_edge!(builder, id, incoming, outgoing)
    edges = builder.incoming
    position = edges.offsets[id] + edges.counts[id]
    position < edges.offsets[id + 1] || throw(
        ArgumentError("More compiled edges than graph degree at node $id")
    )
    edges.slots[position] = incoming
    builder.outgoing.slots[position] = outgoing
    edges.counts[id] += 1
    return nothing
end

function compiled_factor_slots!(builder, neighbors)
    graph, program = getmodel(builder.model), builder.program
    inputs, outputs, marginals = Int32[], Int32[], Int32[]
    for (label, _, node) in neighbors
        variable = getvariable(node)
        if ReactiveMP.israndom(variable)
            initial = getextra(node, InitMsgExtraKey, nothing)
            input = ReactiveMP.compiled_slot!(
                program,
                initial === nothing ? nothing : Message(initial, false, true),
            )
        else
            input = getextra(node, CompiledObservationKey)
        end
        output = if ReactiveMP.isconst(variable)
            Int32(0)
        else
            ReactiveMP.compiled_slot!(program)
        end
        push!(inputs, input)
        push!(outputs, output)
        push!(marginals, getextra(node, CompiledMarginalKey))
        if !iszero(output)
            compiled_append_edge!(builder, label.global_counter, output, input)
        end
    end
    return Tuple(inputs), Tuple(outputs), Tuple(marginals)
end

compiled_tag(names) = isempty(names) ? nothing : Val(Tuple(names))
compiled_bindings(slots) = isempty(slots) ? nothing : Tuple(slots)
compiled_cluster_name(names, cluster) =
    Symbol(join((names[i] for i in cluster), "_"))
compiled_dependency_name(dependency) = ReactiveMP.name(dependency)
compiled_dependency_name(dependencies::Tuple) =
    compiled_dependency_name(first(dependencies))
compiled_dependency_message(dependency::ReactiveMP.CompiledInterface) =
    dependency.message
compiled_dependency_message(dependencies::Tuple) =
    ReactiveMP.CompiledManyOf(map(compiled_dependency_message, dependencies))
compiled_dependency_marginal(dependency::ReactiveMP.CompiledInterface) =
    dependency.marginal
compiled_dependency_marginal(dependency::ReactiveMP.CompiledLocalMarginal) =
    dependency.slot
compiled_dependency_marginal(dependencies::Tuple) =
    ReactiveMP.CompiledManyOf(map(compiled_dependency_marginal, dependencies))

struct CompiledStochasticScore{F, N, M, T}
    fform::F
    names::N
    meta::M
    type::Type{T}
end
function (kernel::CompiledStochasticScore)(marginals)
    energy = ReactiveMP.score(
        ReactiveMP.AverageEnergy(),
        kernel.fform,
        kernel.names,
        marginals,
        kernel.meta,
    )
    entropy = sum(compiled_entropy, marginals)
    return convert(kernel.type, energy - entropy)
end
compiled_entropy(m) = ReactiveMP.score(ReactiveMP.DifferentialEntropy(), m)
compiled_entropy(ms::ReactiveMP.ManyOf) = sum(compiled_entropy, ms)

struct CompiledEntropyScore{T}
    degree::Int
    type::Type{T}
end

struct CompiledDeterministicScore{F, V, N, M, T}
    fform::F
    tag::V
    names::N
    meta::M
    type::Type{T}
end
function (kernel::CompiledDeterministicScore)(messages)
    marginal = Marginal(
        ReactiveMP.marginalrule(
            kernel.fform,
            kernel.tag,
            kernel.names,
            messages,
            nothing,
            nothing,
            kernel.meta,
            nothing,
        ),
        false,
        false,
    )
    return convert(
        kernel.type,
        -ReactiveMP.score(ReactiveMP.DifferentialEntropy(), marginal),
    )
end

struct CompiledJointEntropyScore{T}
    type::Type{T}
end
(kernel::CompiledJointEntropyScore)(marginal) = convert(
    kernel.type,
    -ReactiveMP.score(ReactiveMP.DifferentialEntropy(), marginal),
)
function (kernel::CompiledEntropyScore)(marginal)
    scale = ReactiveMP.ispointmass(marginal) ? kernel.degree : kernel.degree - 1
    return scale * convert(
        kernel.type,
        ReactiveMP.score(ReactiveMP.DifferentialEntropy(), marginal),
    )
end

"""
    lower_compiled_factor!(builder, fform, label, nodedata)

Lower a factor to indexed message/marginal operations. Custom imperative node
layouts must implement this entry point; it must not instantiate reactive nodes.
"""
function lower_compiled_factor!(builder, fform, label, node)
    if fform isa Function &&
        ReactiveMP.is_predefined_node(fform) isa
       ReactiveMP.UndefinedNodeFunctionalForm
        return lower_compiled_delta!(builder, fform, label, node)
    end
    properties = getproperties(node)
    neighbors = GraphPPL.neighbors(properties)
    names = Tuple(
        ReactiveMP.alias_interface(fform, i, GraphPPL.getname(edge)) for
        (i, (_, edge, _)) in enumerate(neighbors)
    )
    length(unique(names)) == length(names) || throw(
        UnsupportedCompiledFeature(
            "grouped interfaces for $fform (a compact layout adapter is required)",
        ),
    )
    constraints = getextra(
        node, GraphPPL.VariationalConstraintsFactorizationIndicesKey
    )
    ReactiveMP.is_predefined_node(fform) isa
    ReactiveMP.UndefinedNodeFunctionalForm &&
        throw(UnsupportedCompiledFeature("custom node context for $fform"))
    clusters = ReactiveMP.collect_factorisation(fform, constraints)
    clusters isa Tuple || throw(
        UnsupportedCompiledFeature("custom factorization layout for $fform")
    )
    meta = ReactiveMP.collect_meta(
        fform, getextra(node, GraphPPL.MetaExtraKey, nothing)
    )
    dependencies = ReactiveMP.collect_functional_dependencies(
        fform, getextra(node, ReactiveMPExtraDependenciesKey, nothing)
    )
    dependencies isa ReactiveMP.FunctionalDependencies || throw(
        UnsupportedCompiledFeature(
            "dependency policy $(typeof(dependencies)) for $fform"
        ),
    )
    getextra(node, ReactiveMPExtraStreamPostprocessorsKey, nothing) ===
    nothing || throw(
        UnsupportedCompiledFeature("per-node stream postprocessors at $label"),
    )
    inputs, outputs, marginals = compiled_factor_slots!(builder, neighbors)
    program = builder.program
    cluster_names = map(
        cluster -> compiled_cluster_name(names, cluster), clusters
    )
    cluster_slots = map(
        cluster -> if length(cluster) == 1
            marginals[first(cluster)]
        else
            ReactiveMP.compiled_slot!(program)
        end, clusters
    )
    interfaces = ntuple(
        i -> ReactiveMP.CompiledInterface(
            names[i], getvariable(neighbors[i][3]), inputs[i], marginals[i]
        ),
        length(names),
    )
    localmarginals = map(
        (name, slot) -> ReactiveMP.CompiledLocalMarginal(name, slot),
        cluster_names,
        cluster_slots,
    )
    layout = ReactiveMP.FactorNode(
        fform,
        interfaces,
        ReactiveMP.FactorNodeLocalClusters(localmarginals, clusters),
    )
    rulecontext = ReactiveMP.node_if_required(fform, layout)
    for (ci, cluster) in enumerate(clusters)
        if length(cluster) > 1
            other = Tuple(i for i in eachindex(clusters) if i != ci)
            mapping = ReactiveMP.MarginalMapping(
                fform,
                Val(cluster_names[ci]),
                compiled_tag(names[collect(cluster)]),
                compiled_tag(cluster_names[collect(other)]),
                meta,
                rulecontext,
            )
            ReactiveMP.compiled_operation!(
                program,
                ReactiveMP.CompiledMarginalKernel(mapping),
                cluster_slots[ci],
                (
                    compiled_bindings(inputs[collect(cluster)]),
                    compiled_bindings(cluster_slots[collect(other)]),
                ),
            )
        end
    end
    for i in eachindex(names)
        iszero(outputs[i]) && continue
        ci = findfirst(cluster -> i in cluster, clusters)
        cluster = clusters[ci]
        include_self =
            dependencies isa
            ReactiveMP.RequireEverythingFunctionalDependencies || (
                dependencies isa
                ReactiveMP.RequireMessageFunctionalDependencies &&
                haskey(dependencies.specification, names[i])
            )
        message_indices = Tuple(j for j in cluster if j != i || include_self)
        other = Tuple(
            j for j in eachindex(clusters) if j != ci ||
                dependencies isa
                ReactiveMP.RequireEverythingFunctionalDependencies
        )
        qnames = Symbol[cluster_names[j] for j in other]
        qslots = Int32[cluster_slots[j] for j in other]
        message_names = names[collect(message_indices)]
        message_bindings = compiled_bindings(inputs[collect(message_indices)])
        marginal_bindings = nothing
        if dependencies isa ReactiveMP.RequireMessageFunctionalDependencies &&
            haskey(dependencies.specification, names[i])
            init = dependencies.specification[names[i]]
            init === nothing ||
                (program.values[inputs[i]] = Message(init, false, true))
        elseif dependencies isa
               ReactiveMP.RequireMarginalFunctionalDependencies &&
            haskey(dependencies.specification, names[i])
            insertat = count(j -> first(clusters[j]) < i, other) + 1
            insert!(qnames, insertat, names[i])
            insert!(qslots, insertat, marginals[i])
            init = dependencies.specification[names[i]]
            init === nothing ||
                (program.values[marginals[i]] = Marginal(init, false, true))
        end
        if !(
            dependencies isa Union{
                ReactiveMP.RequireMessageFunctionalDependencies,
                ReactiveMP.RequireMarginalFunctionalDependencies,
            }
        )
            applicable(
                ReactiveMP.functional_dependencies,
                dependencies,
                layout,
                interfaces[i],
                i,
            ) || throw(
                UnsupportedCompiledFeature(
                    "dependency policy $(typeof(dependencies)) requires a compact layout adapter",
                ),
            )
            mdeps, qdeps = ReactiveMP.functional_dependencies(
                dependencies, layout, interfaces[i], i
            )
            mdeps, qdeps = Tuple(mdeps), Tuple(qdeps)
            message_names = map(compiled_dependency_name, mdeps)
            message_bindings = compiled_bindings(
                map(compiled_dependency_message, mdeps)
            )
            qnames = map(compiled_dependency_name, qdeps)
            marginal_bindings = compiled_bindings(
                map(compiled_dependency_marginal, qdeps)
            )
        else
            marginal_bindings = compiled_bindings(qslots)
        end
        mapping = ReactiveMP.MessageMapping(
            fform,
            Val(names[i]),
            Marginalisation(),
            compiled_tag(message_names),
            compiled_tag(qnames),
            meta,
            getannotations(builder.options),
            rulecontext,
            getrulefallback(builder.options),
            compiled_numerical_callbacks(builder.options),
        )
        ReactiveMP.compiled_operation!(
            program,
            ReactiveMP.CompiledMessageKernel(mapping),
            outputs[i],
            (message_bindings, marginal_bindings),
        )
    end
    if builder.score_type !== nothing
        slot = ReactiveMP.compiled_slot!(program)
        if ReactiveMP.sdtype(fform) isa ReactiveMP.Stochastic
            kernel = CompiledStochasticScore(
                fform, Val(cluster_names), meta, builder.score_type
            )
            ReactiveMP.compiled_operation!(program, kernel, slot, cluster_slots)
        else
            kernel = CompiledDeterministicScore(
                fform,
                Val(compiled_cluster_name(names, 2:length(names))),
                Val(names),
                meta,
                builder.score_type,
            )
            ReactiveMP.compiled_operation!(program, kernel, slot, inputs)
        end
        push!(builder.score_slots, slot)
    end
    return nothing
end

function lower_compiled_delta!(
    builder, fn::F, label, node
) where {F <: Function}
    program = builder.program
    neighbors = GraphPPL.neighbors(getproperties(node))
    meta = ReactiveMP.collect_meta(
        ReactiveMP.DeltaFn{F}, getextra(node, GraphPPL.MetaExtraKey, nothing)
    )
    meta isa ReactiveMP.DeltaMeta ||
        throw(UnsupportedCompiledFeature("Delta metadata $(typeof(meta))"))
    getextra(node, ReactiveMPExtraStreamPostprocessorsKey, nothing) ===
    nothing || throw(
        UnsupportedCompiledFeature("per-node stream postprocessors at $label"),
    )
    inputs, outputs, marginals = compiled_factor_slots!(builder, neighbors)
    random_indices, static_indices, arguments = Int[], Int[], Int[]
    for i in 2:length(neighbors)
        if ReactiveMP.israndom(getvariable(neighbors[i][3]))
            push!(random_indices, i)
            push!(arguments, length(random_indices))
        else
            push!(static_indices, i)
            push!(arguments, -length(static_indices))
        end
    end
    isempty(random_indices) && throw(
        UnsupportedCompiledFeature(
            "Delta node without random inputs at $label"
        ),
    )
    # Static arguments have no Delta message interface in the reference
    # backend, hence contribute no cancelling point-entropy term either.
    builder.point_entropy_count -= length(static_indices)
    statics = Tuple(inputs[i] for i in static_indices)
    random_inputs = Tuple(inputs[i] for i in random_indices)
    grouped = ReactiveMP.CompiledManyOf(random_inputs)
    joint = ReactiveMP.compiled_slot!(program)
    jointmapping = ReactiveMP.MarginalMapping(
        ReactiveMP.DeltaFn{F},
        Val(:ins),
        Val((:out, :ins)),
        nothing,
        meta,
        nothing,
    )
    jointkernel = ReactiveMP.CompiledDeltaKernel(
        fn, Tuple(arguments), jointmapping
    )
    ReactiveMP.compiled_operation!(
        program, jointkernel, joint, (((inputs[1], grouped), nothing), statics)
    )
    function message!(output, tag, mnames, messages, qnames, qs)
        iszero(output) && return nothing
        mapping = ReactiveMP.MessageMapping(
            ReactiveMP.DeltaFn{F},
            tag,
            Marginalisation(),
            mnames,
            qnames,
            meta,
            getannotations(builder.options),
            nothing,
            getrulefallback(builder.options),
            compiled_numerical_callbacks(builder.options),
        )
        kernel = ReactiveMP.CompiledDeltaKernel(fn, Tuple(arguments), mapping)
        ReactiveMP.compiled_operation!(
            program, kernel, output, ((messages, qs), statics)
        )
    end
    if ReactiveMP.getmethod(meta) isa ReactiveMP.CVI
        message!(
            outputs[1], Val(:out), nothing, nothing, Val((:ins,)), (joint,)
        )
    else
        message!(
            outputs[1], Val(:out), Val((:ins,)), (grouped,), nothing, nothing
        )
    end
    for (k, i) in enumerate(random_indices)
        if ReactiveMP.getinverse(meta) === nothing
            message!(
                outputs[i],
                (Val(:in), k),
                Val((:in,)),
                (inputs[i],),
                Val((:ins,)),
                (joint,),
            )
        else
            others = Tuple(inputs[j] for j in random_indices if j != i)
            other_binding = if isempty(others)
                ReactiveMP.compiled_slot!(program, Message(nothing, true, true))
            else
                ReactiveMP.CompiledManyOf(others)
            end
            message!(
                outputs[i],
                (Val(:in), k),
                Val((:out, :ins)),
                (inputs[1], other_binding),
                nothing,
                nothing,
            )
        end
    end
    if builder.score_type !== nothing
        slot = ReactiveMP.compiled_slot!(program)
        ReactiveMP.compiled_operation!(
            program, CompiledJointEntropyScore(builder.score_type), slot, joint
        )
        push!(builder.score_slots, slot)
    end
    return nothing
end

function compiled_lower_variables!(builder; required = Int32[])
    graph, program = getmodel(builder.model), builder.program
    referenced = falses(length(program.values))
    foreach(slot -> referenced[slot] = true, required)
    for operation in program.operations
        ReactiveMP.foreach_compiled_slot(
            slot -> referenced[slot] = true, operation.inputs
        )
    end
    variable_nodes(graph) do label, node
        variable = getvariable(node)
        incoming = builder.incoming[label.global_counter]
        if ReactiveMP.israndom(variable)
            inputs = length(incoming) <= 16 ? Tuple(incoming) : incoming
            outgoing = builder.outgoing[label.global_counter]
            messages_kernel = ReactiveMP.CompiledProductKernel(
                variable,
                compiled_context(builder.options, node, :messages),
                false,
            )
            marginal_kernel = ReactiveMP.CompiledProductKernel(
                variable,
                compiled_context(builder.options, node, :marginal),
                true,
            )
            for i in eachindex(outgoing)
                referenced[outgoing[i]] || continue
                selected = if length(inputs) <= 16
                    Tuple(inputs[j] for j in eachindex(inputs) if j != i)
                else
                    ReactiveMP.CompiledExcept(inputs, i)
                end
                ReactiveMP.compiled_operation!(
                    program, messages_kernel, outgoing[i], selected
                )
            end
            marginal = getextra(node, CompiledMarginalKey)
            ReactiveMP.compiled_operation!(
                program, marginal_kernel, marginal, inputs
            )
            if builder.score_type !== nothing
                slot = ReactiveMP.compiled_slot!(program)
                ReactiveMP.compiled_operation!(
                    program,
                    CompiledEntropyScore(
                        ReactiveMP.degree(variable), builder.score_type
                    ),
                    slot,
                    marginal,
                )
                push!(builder.score_slots, slot)
            end
        elseif ReactiveMP.isdata(variable)
            if incoming !== nothing
                kernel = ReactiveMP.CompiledProductKernel(
                    variable, ReactiveMP.MessageProductContext(), true
                )
                ReactiveMP.compiled_operation!(
                    program,
                    kernel,
                    getextra(node, CompiledPredictionKey),
                    Tuple(incoming),
                )
            end
            link = GraphPPL.value(getproperties(node))
            if link !== nothing
                transform, args = link
                slots = Int32[]
                function lower_arg(arg)
                    if GraphPPL.is_nodelabel(arg)
                        push!(slots, getextra(graph[arg], CompiledMarginalKey))
                        return length(slots)
                    elseif arg isa AbstractArray &&
                        any(GraphPPL.is_nodelabel, arg)
                        return map(lower_arg, arg)
                    end
                    return ReactiveMP.CompiledLiteral(arg)
                end
                kernel = ReactiveMP.CompiledLinkKernel(
                    transform, map(lower_arg, args)
                )
                ReactiveMP.compiled_operation!(
                    program,
                    kernel,
                    getextra(node, CompiledObservationKey),
                    Tuple(slots),
                )
            end
        end
    end
    return builder
end

function compiled_update_data!(model, program, data)
    graph = getmodel(model)
    context = GraphPPL.getcontext(graph)
    for (name, raw) in pairs(data)
        haskey(context, name) || continue
        labels = context[name]
        values = __normalize_data_indexing(get_data(raw))
        if labels isa GraphPPL.NodeLabel
            compiled_update_observation!(graph[labels], program, values)
        else
            for I in CartesianIndices(size(labels))
                isassigned(labels, I.I...) || continue
                compiled_update_observation!(
                    graph[labels[I.I...]], program, values[I]
                )
            end
        end
    end
    return nothing
end

function compiled_update_observation!(node, program, value)
    properties = getproperties(node)
    is_data(properties) || throw(
        ArgumentError(
            "Cannot update non-data variable $(GraphPPL.getname(properties))",
        ),
    )
    GraphPPL.value(properties) === nothing ||
        throw(ArgumentError("Cannot directly update a linked data variable"))
    (
        value isa Union{
            Missing, PointMass, Real, AbstractArray, ReactiveMP.UniformScaling
        }
    ) || throw(
        ArgumentError(
            "Unsupported observation type $(typeof(value)); explicitly wrap non-numeric data in PointMass",
        ),
    )
    payload = value isa Union{Missing, PointMass} ? value : PointMass(value)
    program.values[getextra(node, CompiledObservationKey)] = Message(
        payload, false, false
    )
    return nothing
end

function compiled_selection(model, spec, prediction, iterations)
    variables = GraphPPL.variables(getvardict(model))
    predicate = prediction ? ReactiveMP.isdata : ReactiveMP.israndom
    if spec === nothing || spec isa Union{KeepEach, KeepLast}
        keep = something(spec, iterations === nothing ? KeepLast() : KeepEach())
        return Dict(
            name => keep for (name, refs) in pairs(variables) if
            predicate(refs) && !isanonymous(refs)
        )
    end
    return Dict(pairs(spec))
end

function compiled_selected_slots(model, selection, prediction)
    graph = getmodel(model)
    variables = GraphPPL.variables(getvardict(model))
    key = prediction ? CompiledPredictionKey : CompiledMarginalKey
    function resolve(ref::GraphVariableRef)
        return getextra(graph[getlabel(ref)], key)
    end
    resolve(refs::AbstractArray) = _map_sparse_or_dense(resolve, refs)
    return Dict(name => resolve(variables[name]) for name in keys(selection))
end
_map_sparse_or_dense(f, refs::GraphPPL.ResizableArray) = _map_sparse(f, refs)
_map_sparse_or_dense(f, refs::AbstractArray) = map(f, refs)

compiled_snapshot(program, slot::Int32) = program.values[slot]
compiled_snapshot(program, slots::AbstractArray) =
    _map_sparse_or_dense(slot -> compiled_snapshot(program, slot), slots)
compiled_has_result(value) = value !== nothing
compiled_has_result(values::AbstractArray) =
    all(compiled_has_result, vec(values))

function compiled_model_builder(
    model, data, initialization, constraints, meta, config, runner, CT
)
    plugins = GraphPPL.PluginsCollection(
        GraphPPL.CompactGraphPlugin(),
        GraphPPL.VariationalConstraintsPlugin(constraints),
        GraphPPL.MetaPlugin(meta),
        InitializationPlugin(initialization),
        CompiledOptionsPlugin(),
    )
    generator = GraphPPL.with_backend(
        GraphPPL.with_plugins(model, plugins),
        ReactiveMPGraphPPLBackend(Static.static(false)),
    )
    callbacks = getcallbacks(config)
    creation_span = generate_span_id(callbacks)
    invoke_callback(callbacks, BeforeModelCreationEvent(creation_span))
    fmodel = create_model(generator | data)
    @debug "Compiled construction: GraphPPL complete" peak_rss = Sys.maxrss() live_heap = Base.gc_live_bytes()
    graph = getmodel(fmodel)
    program = ReactiveMP.CompiledProgram(runner)
    incoming, outgoing = compiled_edge_bindings(graph)
    builder = CompiledModelBuilder(
        fmodel, program, config, incoming, outgoing, Int32[], CT, 0
    )
    compiled_prepare_variables!(builder)
    @debug "Compiled construction: variables prepared" peak_rss = Sys.maxrss() live_heap = Base.gc_live_bytes()
    factor_nodes(graph) do label, node
        lower_compiled_factor!(
            builder, GraphPPL.fform(getproperties(node)), label, node
        )
    end
    @debug "Compiled construction: factors lowered" peak_rss = Sys.maxrss() live_heap = Base.gc_live_bytes()
    fmodel.metadata[:inference_runner] = program
    fmodel.metadata[:execution_backend] = :compiled
    return builder, creation_span
end

function compiled_finalize!(builder, roots)
    @debug "Compiled construction: variables lowered" peak_rss = Sys.maxrss() live_heap = Base.gc_live_bytes()
    # All edge bindings now belong to operations (and streaming fetch handles).
    # Release the lowering worklists before allocating the dependency graph.
    # Merely empty! would keep both node-sized pointer arrays' spare capacity.
    builder.incoming = CompiledEdgeBindings()
    builder.outgoing = CompiledEdgeBindings()
    if getforce_marginal_computation(builder.options)
        variable_nodes(getmodel(builder.model)) do _, node
            push!(roots, getextra(node, CompiledMarginalKey))
        end
    end
    ReactiveMP.prune_compiled_operations!(builder.program, roots)
    ReactiveMP.fuse_compiled_variational_products!(builder.program, roots)
    # At very large sizes, construction and scheduling have disjoint temporary
    # heaps. Collect at this phase boundary rather than retaining both peaks.
    large = length(builder.program.operations) >= 100_000
    large && GC.gc(true)
    @debug "Compiled construction: liveness complete" peak_rss = Sys.maxrss() live_heap = Base.gc_live_bytes()
    GraphPPL.compact_shrink!(getmodel(builder.model))
    for values in (
        builder.program.operations,
        builder.program.kernels,
        builder.program.values,
    )
        sizehint!(values, length(values); shrink = true)
    end
    large && GC.gc(true)
    @debug "Compiled construction: storage shrunk" peak_rss = Sys.maxrss() live_heap = Base.gc_live_bytes()
    ReactiveMP.compile_schedule!(builder.program)
    large && GC.gc(true)
    @debug "Compiled construction: schedule complete" peak_rss = Sys.maxrss() live_heap = Base.gc_live_bytes()
    return builder
end

compiled_check_score(::Nothing, value) = nothing
compiled_check_score(checks::Tuple, value) =
    foreach(check -> compiled_check_score(check, value), checks)
function compiled_check_score(::ObjectiveDiagnosticCheckNaNs, value)
    isnan(value) && throw(ArgumentError("NaN compiled Bethe free energy"))
end
function compiled_check_score(::ObjectiveDiagnosticCheckInfs, value)
    isinf(value) && throw(ArgumentError("Infinite compiled Bethe free energy"))
end
function compiled_free_energy(
    program, slots, T, point_entropy_count, diagnostics
)
    all(slot -> program.values[slot] !== nothing, slots) ||
        throw(ArgumentError("Free-energy dependencies remain unresolved"))
    value =
        sum(
            slot -> program.values[slot],
            slots;
            init = convert(__as_counting_real_type(T), 0.0),
        ) - CountingReal(T, point_entropy_count)
    value = float(value)
    compiled_check_score(diagnostics, value)
    return value
end

function compiled_batch_inference(;
    model,
    data,
    initialization,
    constraints,
    meta,
    options,
    returnvars,
    predictvars,
    iterations,
    free_energy,
    free_energy_diagnostics,
    allow_node_contraction,
    showprogress,
    callbacks,
    annotations,
    postprocess,
    warn,
    catch_exception,
    disable_inference_error_hint,
)
    allow_node_contraction &&
        throw(UnsupportedCompiledFeature("node contraction"))
    haskey(options, :stream_postprocessors) &&
        throw(UnsupportedCompiledFeature("stream postprocessors"))
    cleaned_options = (;
        (
            k => v for
            (k, v) in pairs(options) if k ∉ (:runner, :limit_stack_depth)
        )...
    )
    config = convert(ReactiveMPInferenceOptions, cleaned_options)
    callbacks = callbacks === nothing ? getcallbacks(config) : callbacks
    config = setcallbacks(config, callbacks)
    annotations === nothing || (config = setannotations(config, annotations))
    postprocess = something(
        postprocess,
        if getannotations(config) === nothing
            UnpackMarginalPostprocess()
        else
            NoopPostprocess()
        end,
    )
    is_fe, T = unwrap_free_energy_option(free_energy)
    CT = is_fe ? __as_counting_real_type(T) : nothing
    default_prediction = iterations === nothing ? KeepLast() : KeepEach()
    if predictvars isa Union{KeepEach, KeepLast}
        data === nothing && throw(
            ArgumentError(
                "Supply data or explicit variable names with predictvars"
            ),
        )
        predictvars = Dict(name => predictvars for name in keys(data))
    end
    data = if data === nothing
        Dict{Symbol, Any}(
            name => missing for name in keys(something(predictvars, (;)))
        )
    else
        data
    end
    isempty(data) &&
        throw(ArgumentError("Data is empty; supply data or named predictvars"))
    predictions_requested = Dict{Symbol, Any}(
        pairs(something(predictvars, (;)))
    )
    for name in keys(predictions_requested)
        haskey(data, name) || (data = merge(data, Dict(name => missing)))
    end
    for (name, value) in pairs(data)
        if inference_check_dataismissing(get_data(value)) &&
            !haskey(predictions_requested, name)
            predictions_requested[name] = default_prediction
        end
    end
    predictvars = predictions_requested
    builder, creation_span = compiled_model_builder(
        model,
        data,
        initialization,
        constraints,
        meta,
        config,
        options.runner,
        CT,
    )
    fmodel, program = builder.model, builder.program
    compiled_lower_variables!(builder)
    returns = compiled_selection(fmodel, returnvars, false, iterations)
    predictions = compiled_selection(fmodel, predictvars, true, iterations)
    returnslots = compiled_selected_slots(fmodel, returns, false)
    predictslots = compiled_selected_slots(fmodel, predictions, true)
    roots = copy(builder.score_slots)
    for slots in Iterators.flatten((values(returnslots), values(predictslots)))
        slots isa Int32 ? push!(roots, slots) : append!(roots, vec(slots))
    end
    compiled_finalize!(builder, roots)
    invoke_callback(callbacks, AfterModelCreationEvent(fmodel, creation_span))
    niterations = something(iterations, 1)
    niterations isa Integer && niterations > 0 ||
        throw(ArgumentError("iterations must be a positive integer"))
    posterior_values, prediction_values = Dict{Symbol, Any}(),
    Dict{Symbol, Any}()
    for (name, keep) in pairs(returns)
        posterior_values[name] = keep isa KeepEach ? Any[] : missing
    end
    for (name, keep) in pairs(predictions)
        prediction_values[name] = keep isa KeepEach ? Any[] : missing
    end
    scores = is_fe ? T[] : nothing
    potential_error = nothing
    inference_span = generate_span_id(callbacks)
    try
        invoke_callback(callbacks, BeforeInferenceEvent(fmodel, inference_span))
        for iteration in 1:niterations
            span = generate_span_id(callbacks)
            before = BeforeIterationEvent(fmodel, iteration, span)
            invoke_callback(callbacks, before)
            before.stop_iteration && break
            data_span = generate_span_id(callbacks)
            invoke_callback(
                callbacks, BeforeDataUpdateEvent(fmodel, data, data_span)
            )
            compiled_update_data!(fmodel, program, data)
            ReactiveMP.compiled_sweep!(program)
            invoke_callback(
                callbacks, AfterDataUpdateEvent(fmodel, data, data_span)
            )
            for (selection, slots, output) in (
                (returns, returnslots, posterior_values),
                (predictions, predictslots, prediction_values),
            )
                for (name, keep) in pairs(selection)
                    value = compiled_snapshot(program, slots[name])
                    compiled_has_result(value) || throw(
                        ArgumentError(
                            "Compiled inference has unresolved dependencies for $name; check initialization",
                        ),
                    )
                    invoke_callback(
                        callbacks, OnMarginalUpdateEvent(fmodel, name, value)
                    )
                    value = inference_postprocess(postprocess, value)
                    if keep isa KeepEach
                        push!(output[name], deepcopy(value))
                    else
                        (output[name] = value)
                    end
                end
            end
            if is_fe
                value = compiled_free_energy(
                    program,
                    builder.score_slots,
                    T,
                    builder.point_entropy_count,
                    free_energy_diagnostics,
                )
                push!(scores, value)
            end
            after = AfterIterationEvent(fmodel, iteration, span)
            invoke_callback(callbacks, after)
            after.stop_iteration && break
        end
        invoke_callback(callbacks, AfterInferenceEvent(fmodel, inference_span))
    catch error
        program.failed = true
        potential_error = inference_process_error(
            error; rethrow = !catch_exception, disable_inference_error_hint
        )
    end
    return InferenceResult(
        posterior_values, prediction_values, scores, fmodel, potential_error
    )
end
