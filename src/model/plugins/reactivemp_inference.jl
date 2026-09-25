import GraphPPL:
    plugin_type,
    FactorAndVariableNodesPlugin,
    preprocess_plugin,
    postprocess_plugin
import GraphPPL:
    Model,
    Context,
    NodeLabel,
    NodeData,
    FactorNodeProperties,
    VariableNodeProperties,
    NodeCreationOptions,
    hasextra,
    getextra,
    setextra!,
    getproperties
import GraphPPL: as_variable, is_data, is_random, is_constant, degree
import GraphPPL: variable_nodes, factor_nodes

"""
    ReactiveMPInferenceOptions(; kwargs...)

Creates model inference options object. The list of available options is present below.

### Options

- `limit_stack_depth`: limits the stack depth for computing messages; helps with `StackOverflowError` for some huge models, but reduces the performance of the inference backend. Accepts an integer argument that specifies the maximum recursion depth. Lower is better for stack overflow errors, but worse for performance.
- `warn`: (optional) flag to suppress warnings. Warnings are not displayed if set to `false`. Defaults to `true`.
- `force_marginal_computation`: (optional) flag to force computation of marginals even when not explicitly requested. Defaults to `false`.

### Advanced options

- `stream_postprocessors`: changes the postprocessor of reactive streams, see ReactiveMP.jl for more info, defaults to `nothing`, unless the `limit_stack_depth` option is set, in which case will be set to `ReactiveMP.ScheduleOnStreamPostprocessor` together with `RxInfer.LimitStackScheduler`.
- `diagnostics`: the engine's audits of the rules the model runs, a `ReactiveMP.EngineDiagnostics` (`check_everything_pure`, `check_everything_inplace`, `checked_buffers`), all off by default.
- `rng`: the random number generator the rules draw from; the task's own by default.

v6's `rulefallback` is gone: when no rule fits, the engine reports the closest candidates.

See also: [`infer`](@ref)
"""
struct ReactiveMPInferenceOptions{S, A, R, E, G}
    stream_postprocessors::S
    annotations::A
    warn::Bool
    force_marginal_computation::Bool
    diagnostics::R
    callbacks::E
    rng::G
end

ReactiveMPInferenceOptions(stream_postprocessors, annotations) =
    ReactiveMPInferenceOptions(
        stream_postprocessors, annotations, true, false, nothing, nothing, nothing
    )
ReactiveMPInferenceOptions(stream_postprocessors, annotations, warn) =
    ReactiveMPInferenceOptions(
        stream_postprocessors, annotations, warn, false, nothing, nothing, nothing
    )
ReactiveMPInferenceOptions(
    stream_postprocessors, annotations, warn, force_marginal_computation
) = ReactiveMPInferenceOptions(
    stream_postprocessors,
    annotations,
    warn,
    force_marginal_computation,
    nothing,
    nothing,
    nothing,
)
ReactiveMPInferenceOptions(
    stream_postprocessors,
    annotations,
    warn,
    force_marginal_computation,
    diagnostics,
) = ReactiveMPInferenceOptions(
    stream_postprocessors,
    annotations,
    warn,
    force_marginal_computation,
    diagnostics,
    nothing,
    nothing,
)
ReactiveMPInferenceOptions(
    stream_postprocessors,
    annotations,
    warn,
    force_marginal_computation,
    diagnostics,
    callbacks,
) = ReactiveMPInferenceOptions(
    stream_postprocessors,
    annotations,
    warn,
    force_marginal_computation,
    diagnostics,
    callbacks,
    nothing,
)

setpostprocessor(options::ReactiveMPInferenceOptions, stream_postprocessors) =
    ReactiveMPInferenceOptions(
        stream_postprocessors,
        options.annotations,
        options.warn,
        options.force_marginal_computation,
        options.diagnostics,
        options.callbacks,
        options.rng,
    )
setannotations(options::ReactiveMPInferenceOptions, annotations) =
    ReactiveMPInferenceOptions(
        options.stream_postprocessors,
        annotations,
        options.warn,
        options.force_marginal_computation,
        options.diagnostics,
        options.callbacks,
        options.rng,
    )
setwarn(options::ReactiveMPInferenceOptions, warn) = ReactiveMPInferenceOptions(
    options.stream_postprocessors,
    options.annotations,
    warn,
    options.force_marginal_computation,
    options.diagnostics,
    options.callbacks,
    options.rng,
)
setforce_marginal_computation(
    options::ReactiveMPInferenceOptions, force_marginal_computation
) = ReactiveMPInferenceOptions(
    options.stream_postprocessors,
    options.annotations,
    options.warn,
    force_marginal_computation,
    options.diagnostics,
    options.callbacks,
    options.rng,
)
setdiagnostics(options::ReactiveMPInferenceOptions, diagnostics) =
    ReactiveMPInferenceOptions(
        options.stream_postprocessors,
        options.annotations,
        options.warn,
        options.force_marginal_computation,
        diagnostics,
        options.callbacks,
        options.rng,
    )
setcallbacks(options::ReactiveMPInferenceOptions, callbacks) =
    ReactiveMPInferenceOptions(
        options.stream_postprocessors,
        options.annotations,
        options.warn,
        options.force_marginal_computation,
        options.diagnostics,
        callbacks,
        options.rng,
    )

import Base: convert

function Base.convert(::Type{ReactiveMPInferenceOptions}, options::Nothing)
    return convert(ReactiveMPInferenceOptions, (;))
end

function Base.convert(
    ::Type{ReactiveMPInferenceOptions}, options::NamedTuple{keys}
) where {keys}
    available_options = (
        :stream_postprocessors,
        :limit_stack_depth,
        :annotations,
        :warn,
        :diagnostics,
        :force_marginal_computation,
        :callbacks,
        :rng,
    )

    :rulefallback in keys && error(
        "The `rulefallback` option is gone in ReactiveMP v7: when no rule fits, the engine reports the closest candidates. See the ReactiveMP v6 → v7 migration guide.",
    )
    for key in keys
        key ∈ available_options || error(
            "Unknown model inference options: $(key). Available options are: $(available_options). ",
        )
    end

    warn = haskey(options, :warn) ? options.warn : true
    annotations = haskey(options, :annotations) ? options.annotations : nothing
    diagnostics =
        haskey(options, :diagnostics) ? options.diagnostics : nothing
    rng = haskey(options, :rng) ? options.rng : nothing
    force_marginal_computation = if haskey(options, :force_marginal_computation)
        options.force_marginal_computation
    else
        false
    end
    callbacks = haskey(options, :callbacks) ? options.callbacks : nothing

    if warn &&
        haskey(options, :stream_postprocessors) &&
        haskey(options, :limit_stack_depth)
        @warn "Inference options have `stream_postprocessors` and `limit_stack_depth` options specified together. Ignoring `limit_stack_depth`. Use `warn = false` option in `ModelInferenceOptions` to suppress this warning."
    end

    stream_postprocessors = if haskey(options, :stream_postprocessors)
        options[:stream_postprocessors]
    elseif haskey(options, :limit_stack_depth)
        ReactiveMP.ScheduleOnStreamPostprocessor(
            LimitStackScheduler(options[:limit_stack_depth]...)
        )
    else
        nothing
    end

    return ReactiveMPInferenceOptions(
        stream_postprocessors,
        annotations,
        warn,
        force_marginal_computation,
        diagnostics,
        callbacks,
        rng,
    )
end

import ReactiveMP:
    getannotations, getcallbacks, getpostprocessor

ReactiveMP.getannotations(options::ReactiveMPInferenceOptions) =
    ReactiveMP.getannotations(options, options.annotations)
ReactiveMP.getannotations(
    options::ReactiveMPInferenceOptions,
    annotations::ReactiveMP.AbstractAnnotations,
) = (annotations,) # ReactiveMP expects annotations to be of type tuple
ReactiveMP.getannotations(
    options::ReactiveMPInferenceOptions, annotations::Nothing
) = annotations                     # Do nothing if annotations is `nothing`
ReactiveMP.getannotations(
    options::ReactiveMPInferenceOptions, annotations::Tuple
) = annotations                       # Do nothing if annotations is a `Tuple`
getdiagnostics(options::ReactiveMPInferenceOptions) =
    something(options.diagnostics, ReactiveMP.EngineDiagnostics())
getrng(options::ReactiveMPInferenceOptions) = options.rng
ReactiveMP.getcallbacks(options::ReactiveMPInferenceOptions) = options.callbacks
ReactiveMP.getpostprocessor(options::ReactiveMPInferenceOptions) =
    options.stream_postprocessors

# Get the force_marginal_computation setting
getforce_marginal_computation(options::ReactiveMPInferenceOptions) =
    options.force_marginal_computation

struct ReactiveMPInferencePlugin{Options <: ReactiveMPInferenceOptions}
    options::Options
end

getoptions(plugin::ReactiveMPInferencePlugin) = plugin.options

const ReactiveMPExtraFactorNodeKey = GraphPPL.NodeDataExtraKey{
    :rmp_factornode, ReactiveMP.AbstractFactorNode
}()
const ReactiveMPExtraVariableKey = GraphPPL.NodeDataExtraKey{
    :rmp_variable, ReactiveMP.AbstractVariable
}()
const ReactiveMPExtraAlgorithmKey = GraphPPL.NodeDataExtraKey{
    :algorithm, ReactiveMP.Any
}()
# The constants a factor node holds that GraphPPL does not know of, such as the distribution
# of a prior `x ~ d`: the free energy cancels their point entropies as it does GraphPPL's.
const ReactiveMPExtraHiddenConstantsKey = GraphPPL.NodeDataExtraKey{
    :hidden_constants, Int
}()
const ReactiveMPExtraStreamPostprocessorsKey = GraphPPL.NodeDataExtraKey{
    :stream_postprocessors, ReactiveMP.Any
}()

GraphPPL.plugin_type(::ReactiveMPInferencePlugin) =
    FactorAndVariableNodesPlugin()

function GraphPPL.preprocess_plugin(
    plugin::ReactiveMPInferencePlugin,
    model::Model,
    context::Context,
    label::NodeLabel,
    nodedata::NodeData,
    options::NodeCreationOptions,
)
    preprocess_plugin(plugin, nodedata, getproperties(nodedata), options)
    return label, nodedata
end

function GraphPPL.preprocess_plugin(
    plugin::ReactiveMPInferencePlugin,
    nodedata::NodeData,
    nodeproperties::VariableNodeProperties,
    options::NodeCreationOptions,
)
    return nothing
end

function GraphPPL.preprocess_plugin(
    plugin::ReactiveMPInferencePlugin,
    nodedata::NodeData,
    nodeproperties::FactorNodeProperties,
    options::NodeCreationOptions,
)
    haskey(options, :dependencies) && error(
        "`where { dependencies = … }` is gone in ReactiveMP v7: a node declares what its rules read (`@define_factor_node`'s `dependencies`, or `@define_dependencies` for an algorithm), and an initial message is set with `@initialization`. See the ReactiveMP v6 → v7 migration guide.",
    )
    if haskey(options, :meta)
        Base.depwarn("`where { meta = … }` is deprecated: a node's meta is its algorithm in ReactiveMP v7, so write `where { algorithm = … }`.", :meta; force = true)
    end
    if haskey(options, GraphPPL.getkey(ReactiveMPExtraAlgorithmKey))
        setextra!(
            nodedata,
            ReactiveMPExtraAlgorithmKey,
            options[GraphPPL.getkey(ReactiveMPExtraAlgorithmKey)],
        )
    end
    if haskey(options, GraphPPL.getkey(ReactiveMPExtraStreamPostprocessorsKey))
        setextra!(
            nodedata,
            ReactiveMPExtraStreamPostprocessorsKey,
            options[GraphPPL.getkey(ReactiveMPExtraStreamPostprocessorsKey)],
        )
    end
    return nothing
end

function GraphPPL.postprocess_plugin(
    plugin::ReactiveMPInferencePlugin, model::Model
)
    # The variable nodes must be instantiated before the factor nodes
    variable_nodes(model) do label, variable
        properties = getproperties(variable)::VariableNodeProperties

        # Additional check for the model, since `ReactiveMP` does not allow half-edges
        if is_random(properties)
            degree(model, label) !== 0 ||
                error(lazy"Unused random variable has been found $(label).")
            degree(model, label) !== 1 || error(
                lazy"Half-edge has been found: $(label). To terminate half-edges 'Uninformative' node can be used.",
            )
        end

        set_rmp_variable!(plugin, model, variable, properties)
    end

    # The nodes must be postprocessed after all variables have been instantiated
    factor_nodes(model) do label, factor
        set_rmp_factornode!(
            plugin, model, factor, getproperties(factor)::FactorNodeProperties
        )
    end

    # The variable nodes must be activated before the factor nodes
    variable_nodes(model) do _, variable
        activate_rmp_variable!(
            plugin,
            model,
            variable,
            getproperties(variable)::VariableNodeProperties,
        )
        if hasextra(variable, InitMarExtraKey)
            ReactiveMP.set_initial_marginal!(
                getextra(variable, ReactiveMPExtraVariableKey),
                getextra(variable, InitMarExtraKey),
            )
        end
        if hasextra(variable, InitMsgExtraKey)
            ReactiveMP.set_initial_message!(
                getextra(variable, ReactiveMPExtraVariableKey),
                getextra(variable, InitMsgExtraKey),
            )
        end
    end

    # The factor nodes must be activated after the variable nodes
    factor_nodes(model) do label, factor
        activate_rmp_factornode!(
            plugin, model, factor, getproperties(factor)::FactorNodeProperties
        )
    end
end

function set_rmp_variable!(
    plugin::ReactiveMPInferencePlugin,
    model::Model,
    nodedata::NodeData,
    nodeproperties::VariableNodeProperties,
)
    varlabel = nodeproperties # TODO: should we make VariableNodeProperties mutable?
    if is_random(nodeproperties)
        return setextra!(
            nodedata, ReactiveMPExtraVariableKey, randomvar(; label = varlabel)
        )
    elseif is_data(nodeproperties)
        return setextra!(
            nodedata, ReactiveMPExtraVariableKey, datavar(; label = varlabel)
        )
    elseif is_constant(nodeproperties)
        return setextra!(
            nodedata,
            ReactiveMPExtraVariableKey,
            constvar(GraphPPL.value(nodeproperties); label = varlabel),
        )
    else
        error(
            "Unknown `kind` in the node properties `$(nodeproperties)` for variable node `$(nodedata)`. Expected `random`, `constant` or `data`.",
        )
    end
end

function activate_rmp_variable!(
    plugin::ReactiveMPInferencePlugin,
    model::Model,
    nodedata::NodeData,
    nodeproperties::VariableNodeProperties,
)
    if is_random(nodeproperties)
        # Fetch "fold-strategy" for messages and marginals. The fold-strategy usually defines the order of messages multiplication (left-to-right)
        # But can use some custom logic for product, e.g. parallel products
        messages_fold_strategy = getextra(
            nodedata,
            :messages_fold_strategy,
            ReactiveMP.MessagesProductFromLeftToRight(),
        )
        marginal_fold_strategy = getextra(
            nodedata,
            :marginal_fold_strategy,
            ReactiveMP.MessagesProductFromLeftToRight(),
        )
        # Fetch "form-constraint" for messages and marginals. The form-constraint usually defines the form of the resulting distribution
        # By default it is `UnspecifiedFormConstraint` which means that the form of the resulting distribution is not specified in advance
        # and follows from the computation, but users may override it with other form constraints, e.g. `PointMassFormConstraint`, which
        # constrains the resulting distribution to be of a point mass form
        messages_form_constraint =
            ReactiveMP.preprocess_form_constraints(
                plugin,
                model,
                getextra(
                    nodedata,
                    GraphPPL.VariationalConstraintsMessagesFormConstraintKey,
                    ReactiveMP.UnspecifiedFormConstraint(),
                ),
            ) + EnsureSupportedFunctionalForm(
                :μ,
                GraphPPL.getname(nodeproperties),
                GraphPPL.index(nodeproperties),
            )
        marginal_form_constraint =
            ReactiveMP.preprocess_form_constraints(
                plugin,
                model,
                getextra(
                    nodedata,
                    GraphPPL.VariationalConstraintsMarginalFormConstraintKey,
                    ReactiveMP.UnspecifiedFormConstraint(),
                ),
            ) + EnsureSupportedFunctionalForm(
                :q,
                GraphPPL.getname(nodeproperties),
                GraphPPL.index(nodeproperties),
            )
        # Fetch "prod-constraint" for messages and marginals. The prod-constraint usually defines the constraints for a single product of messages
        # It can for example preserve a specific parametrization of distribution, see BayesBase.prod documentation for more details on that
        messages_prod_constraint = getextra(
            nodedata,
            :messages_prod_constraint,
            ReactiveMP.default_prod_constraint(messages_form_constraint),
        )
        marginal_prod_constraint = getextra(
            nodedata,
            :marginal_prod_constraint,
            ReactiveMP.default_prod_constraint(marginal_form_constraint),
        )
        # Fetch "form-check-strategy" for messages and marginals. The form-check-strategy usually defines the strategy for checking the form of the resulting distribution
        # The functional form constraint can be applied either after all products are computed or after each product
        messages_form_constraint_check_strategy = getextra(
            nodedata,
            :messages_form_constraint_check_strategy,
            ReactiveMP.default_form_check_strategy(messages_form_constraint),
        )
        marginal_form_constraint_check_strategy = getextra(
            nodedata,
            :marginal_form_constraint_check_strategy,
            ReactiveMP.default_form_check_strategy(marginal_form_constraint),
        )
        # Create the activation options for the random variable which consists of the messages and marginal product functions and stream postprocessor
        prod_context_for_messages_computation = ReactiveMP.MessageProductContext(;
            fold_strategy = messages_fold_strategy,
            prod_constraint = messages_prod_constraint,
            form_constraint = messages_form_constraint,
            form_constraint_check_strategy = messages_form_constraint_check_strategy,
            callbacks = getcallbacks(getoptions(plugin)),
            annotations = getannotations(getoptions(plugin)),
        )
        prod_context_for_marginal_computation = ReactiveMP.MessageProductContext(;
            fold_strategy = marginal_fold_strategy,
            prod_constraint = marginal_prod_constraint,
            form_constraint = marginal_form_constraint,
            form_constraint_check_strategy = marginal_form_constraint_check_strategy,
            callbacks = getcallbacks(getoptions(plugin)),
            annotations = getannotations(getoptions(plugin)),
        )
        options = ReactiveMP.RandomVariableActivationOptions(
            getpostprocessor(getoptions(plugin)),
            prod_context_for_messages_computation,
            prod_context_for_marginal_computation,
        )
        return ReactiveMP.activate!(
            getextra(nodedata, ReactiveMPExtraVariableKey)::RandomVariable,
            options,
        )
    elseif is_data(nodeproperties)
        properties = getproperties(nodedata)::GraphPPL.VariableNodeProperties
        # The datavar can be linked to another variable via a `transform` function, which should be stored in the `value` 
        # field of the properties. In this case the `datavar` gets its values from the linked variable and does not create an explicit factor node
        transform = nothing
        args = nothing
        value = GraphPPL.value(properties)
        if !isnothing(value)
            _transform, _args = value
            transform = _transform
            args = map(arg -> if GraphPPL.is_nodelabel(arg)
                getvariable(getvarref(model, arg))
            else
                arg
            end, _args)
        end
        options = ReactiveMP.DataVariableActivationOptions(
            true, !isnothing(value), transform, args
        )
        return ReactiveMP.activate!(
            getextra(nodedata, ReactiveMPExtraVariableKey)::DataVariable,
            options,
        )
    elseif is_constant(nodeproperties)
        # The constant does not require extra activation
        return nothing
    else
        error(
            "Unknown `kind` in the node properties `$(nodeproperties)` for variable node `$(nodedata)`. Expected `random`, `constant` or `data`.",
        )
    end
end

# A node's algorithm: `where { algorithm = … }`, else what `@algorithm` or v6's `meta`, its
# deprecated spelling, gave it; `nothing` is the node's own default.
function node_algorithm(nodedata::NodeData)
    algorithm = getextra(nodedata, ReactiveMPExtraAlgorithmKey, nothing)
    algorithm = isnothing(algorithm) ? getextra(nodedata, GraphPPL.MetaExtraKey, nothing) : algorithm
    return model_algorithm(GraphPPL.fform(getproperties(nodedata)), algorithm)
end

# A Delta node, a function no package declares, may be given its approximation method alone,
# `f() -> Linearization()`, as v6 allowed; it runs under `DeltaApproximation(method = …)`.
model_algorithm(fform, algorithm) =
    !isdeclarednode(fform) && fform isa Function && is_delta_node_compatible(algorithm) === Val(true) ?
    DeltaApproximation(method = algorithm) : algorithm

# The engine names a node's interfaces, never positions: an interface by its name, a member of
# one of the node's groups as `(name, k)`, `k` being GraphPPL's `EdgeLabel.index`. GraphPPL may
# index an edge that is no group's, such as `out` from a slice of a data array.
interface_key(edge::GraphPPL.EdgeLabel, groups) =
    !isnothing(edge.index) && GraphPPL.getname(edge) in groups ? (GraphPPL.getname(edge), edge.index) : GraphPPL.getname(edge)
interface_key(edge::GraphPPL.EdgeLabel) = isnothing(edge.index) ? GraphPPL.getname(edge) : (GraphPPL.getname(edge), edge.index)

function set_rmp_factornode!(
    plugin::ReactiveMPInferencePlugin,
    model::Model,
    nodedata::NodeData,
    nodeproperties::FactorNodeProperties,
)
    fform = GraphPPL.fform(nodeproperties)
    interfaces = map(GraphPPL.neighbors(nodeproperties)) do (_, edge, data)
        key = isdeclarednode(fform) ? interface_key(edge, MessagePassingRulesBase.interface_groups(fform)) : interface_key(edge)
        return (key, getextra(data, ReactiveMPExtraVariableKey))
    end
    # GraphPPL gives the factorisation as positions in the node's neighbours.
    positions = getextra(
        nodedata, GraphPPL.VariationalConstraintsFactorizationIndicesKey
    )
    node = if isdeclarednode(fform)
        factorization = map(cluster -> map(i -> first(interfaces[i]), Tuple(cluster)), Tuple(positions))
        factornode(fform, interfaces, factorization)
    elseif fform isa Distribution
        # `x ~ d` for a distribution value: the node `out ~ d`, `d` a constant of its own.
        length(interfaces) == 1 || error("A factor node with a distribution object can only have one output interface.")
        (_, variable) = only(interfaces)
        setextra!(nodedata, ReactiveMPExtraHiddenConstantsKey, 1)
        factornode(StandaloneDistribution, [(:out, variable), (:distribution, ReactiveMP.constvar(fform))], ((:out,), (:distribution,)))
    elseif fform isa Function
        delta_factornode(fform, interfaces, positions)
    else
        error("`$(fform)` is not a factor node: no loaded package declares it with `@define_factor_node`")
    end
    return setextra!(nodedata, ReactiveMPExtraFactorNodeKey, node)
end

# A function no package declares is a Delta node, `out = f(in...)`, whose inputs are the group
# `in`: a single input, which GraphPPL leaves unindexed, is its first member.
function delta_factornode(f::F, interfaces, positions) where {F}
    keys = map(interfaces) do (key, _)
        key === :out && return :out
        key isa Symbol && return (:in, 1)
        return (:in, last(key))
    end
    renamed = map((key, (_, variable)) -> (key, variable), keys, interfaces)
    factorization = map(cluster -> map(i -> keys[i], Tuple(cluster)), Tuple(positions))
    return factornode(DeltaFn{F}, renamed, factorization; nodefn = f)
end

function activate_rmp_factornode!(
    plugin::ReactiveMPInferencePlugin,
    model::Model,
    nodedata::NodeData,
    nodeproperties::FactorNodeProperties,
)
    algorithm = node_algorithm(nodedata)
    stream_postprocessors = getextra(
        nodedata, ReactiveMPExtraStreamPostprocessorsKey, nothing
    )
    # if per-node setting is `nothing`, set the global one from the options
    if isnothing(stream_postprocessors)
        stream_postprocessors = getpostprocessor(getoptions(plugin))
    end
    annotations = getannotations(getoptions(plugin))
    callbacks = getcallbacks(getoptions(plugin))

    options = ReactiveMP.FactorNodeActivationOptions(;
        algorithm,
        postprocessor = stream_postprocessors,
        annotations,
        callbacks,
        diagnostics = getdiagnostics(getoptions(plugin)),
        rng = getrng(getoptions(plugin)),
    )

    return ReactiveMP.activate!(
        getextra(nodedata, ReactiveMPExtraFactorNodeKey), options
    )
end

struct GraphVariableRef
    label::GraphPPL.NodeLabel
    properties::GraphPPL.VariableNodeProperties
    variable::ReactiveMP.AbstractVariable
end

getlabel(ref::GraphVariableRef) = ref.label
getvariable(ref::GraphVariableRef) = ref.variable
getname(ref::GraphVariableRef) = GraphPPL.getname(getlabel(ref))

# `vec` (GraphPPL's `ResizableArray` method) yields only the *assigned* entries, so these
# predicates are well-defined for sparse containers (a sparse data tensor is still "all data").
GraphPPL.is_data(collection::AbstractArray{GraphVariableRef}) =
    all(GraphPPL.is_data, vec(collection))

GraphPPL.is_data(ref::GraphVariableRef) = GraphPPL.is_data(ref.properties)
GraphPPL.is_random(ref::GraphVariableRef) = GraphPPL.is_random(ref.properties)
GraphPPL.is_constant(ref::GraphVariableRef) =
    GraphPPL.is_constant(ref.properties)

function GraphVariableRef(model::GraphPPL.Model, label::GraphPPL.NodeLabel)
    nodedata = model[label]::GraphPPL.NodeData
    properties = getproperties(nodedata)::GraphPPL.VariableNodeProperties
    variable = getvariable(nodedata)
    return GraphVariableRef(label, properties, variable)
end

function getreturnval(model::GraphPPL.Model)
    return GraphPPL.returnval(GraphPPL.getcontext(model))
end

function getvardict(model::GraphPPL.Model)
    return map(
        v -> getvarref(model, v), GraphPPL.VarDict(GraphPPL.getcontext(model))
    )
end

getvarref(model::GraphPPL.Model, label::GraphPPL.NodeLabel) =
    GraphVariableRef(model, label)
getvarref(model::GraphPPL.Model, container::AbstractArray) =
    map(element -> getvarref(model, element), container)
getvarref(model::GraphPPL.Model, container::GraphPPL.ResizableArray) =
    _map_sparse(element -> getvarref(model, element), container)

# --- Sparse variable arrays -------------------------------------------------------------
#
# A `GraphPPL.ResizableArray` may be *sparse*: indexed only at some positions within its
# bounding box, leaving the rest as `#undef` holes. This arises for data tensors that are
# conditioned on but only *partially referenced* in the model — e.g. masked / missing
# observations, where a sub-model only touches the observed indices and the remaining
# entries of the conditioned array are never used. GraphPPL's `iterate`/`Base.map` over
# such an array are deliberately dense (so `length`/`collect` stay consistent) and would
# trip over the holes with an `UndefRefError`. The helpers below iterate only the
# *assigned* entries instead, mirroring the sparse-aware `vec`/`isassigned` API.

# Whether every index within a container's bounding box is assigned. Plain `AbstractArray`s
# (and densely-built `ResizableArray`s) are dense; only partially-indexed `ResizableArray`s
# are sparse. Used to keep the common dense path allocation- and behaviour-identical.
_is_densely_assigned(::AbstractArray) = true
_is_densely_assigned(container::GraphPPL.ResizableArray) =
    all(I -> isassigned(container, I.I...), CartesianIndices(size(container)))

# Apply `f` to a (possibly sparse) `ResizableArray`, preserving its shape. Dense arrays take
# the fast path identical to `Base.map` (returning a plain `Array`); sparse arrays map only
# their assigned entries into a new `ResizableArray`, leaving the holes as holes.
function _map_sparse(f::F, container::GraphPPL.ResizableArray) where {F}
    _is_densely_assigned(container) && return map(f, container)
    indices = filter(I -> isassigned(container, I.I...), collect(CartesianIndices(size(container))))
    values  = [f(container[I.I...]) for I in indices]
    result  = GraphPPL.ResizableArray(eltype(values), Val(ndims(container)))
    for (value, I) in zip(values, indices)
        result[I.I...] = value
    end
    return result
end

getvariable(nodedata::GraphPPL.NodeData) =
    getextra(nodedata, ReactiveMPExtraVariableKey)
getvariable(container::AbstractArray) = map(getvariable, container)
getvariable(container::GraphPPL.ResizableArray) =
    _map_sparse(getvariable, container)

# Feed a new observation into a (possibly sparse) array of data variables, aligning by index:
# `datavars[I]` receives `data[I]` for every *assigned* `I`. Entries of the provided `data`
# that the model never referenced are ignored. Dense arrays defer to `ReactiveMP` as before.
new_observation_indexed!(datavars, data) =
    ReactiveMP.new_observation!(datavars, data)
function new_observation_indexed!(
    datavars::GraphPPL.ResizableArray, data::AbstractArray
)
    # The label array is 1-based; normalize offset-indexed `data` so the per-index alignment
    # below (`data[I]`, with 1-based `I`) reads the intended values (see `__normalize_data_indexing`).
    data = __normalize_data_indexing(data)
    _is_densely_assigned(datavars) &&
        return ReactiveMP.new_observation!(datavars, data)
    for I in CartesianIndices(size(datavars))
        if isassigned(datavars, I.I...)
            ReactiveMP.new_observation!(datavars[I.I...], data[I])
        end
    end
    return nothing
end

function getrandomvars(model::GraphPPL.Model)
    # TODO improve performance here
    randomlabels = filter(collect(variable_nodes(model))) do label
        is_random(getproperties(model[label])::GraphPPL.VariableNodeProperties)
    end
    return map(label -> model[label]::GraphPPL.NodeData, randomlabels)
end

function getdatavars(model::GraphPPL.Model)
    # TODO improve performance here
    datalabels = filter(collect(variable_nodes(model))) do label
        is_data(getproperties(model[label])::GraphPPL.VariableNodeProperties)
    end
    return map(label -> model[label]::GraphPPL.NodeData, datalabels)
end

function getconstantvars(model::GraphPPL.Model)
    # TODO improve performance here
    constantlabels = filter(collect(variable_nodes(model))) do label
        is_constant(
            getproperties(model[label])::GraphPPL.VariableNodeProperties
        )
    end
    return map(label -> model[label]::GraphPPL.NodeData, constantlabels)
end

function getfactornodes(model::GraphPPL.Model)
    return map(label -> model[label]::GraphPPL.NodeData, factor_nodes(model))
end

obtain_prediction(ref::GraphVariableRef) =
    ReactiveMP.get_stream_of_predictions(ref.variable) |>
    ReactiveMP.skip_initial()
obtain_prediction(refs::AbstractArray) =
    collectLatest(map(obtain_prediction, refs))

obtain_marginal(ref::GraphVariableRef) =
    ReactiveMP.get_stream_of_marginals(ref.variable) |>
    ReactiveMP.skip_initial()
obtain_marginal(refs::AbstractArray) = collectLatest(map(obtain_marginal, refs))

ReactiveMP.israndom(collection::AbstractArray{GraphVariableRef}) =
    all(ReactiveMP.israndom, vec(collection))
ReactiveMP.isdata(collection::AbstractArray{GraphVariableRef}) =
    all(ReactiveMP.isdata, vec(collection))
ReactiveMP.isconst(collection::AbstractArray{GraphVariableRef}) =
    all(ReactiveMP.isconst, vec(collection))

ReactiveMP.israndom(ref::GraphVariableRef) = GraphPPL.is_random(ref.properties)
ReactiveMP.isdata(ref::GraphVariableRef) = GraphPPL.is_data(ref.properties)
ReactiveMP.isconst(ref::GraphVariableRef) = GraphPPL.is_constant(ref.properties)

isanonymous(collection::AbstractArray{GraphVariableRef}) =
    all(isanonymous, vec(collection))
isanonymous(ref::GraphVariableRef) = GraphPPL.is_anonymous(ref.properties)

# Form constraint preprocessing 

function ReactiveMP.preprocess_form_constraints(
    backend::ReactiveMPInferencePlugin, model::Model, constraints
)
    # It is a simple pass-through for now, but can be extended in the future to preprocess constraints that 
    # are defined in other packages, e.g. in `Distributions` and to support constraints, such as `q(x) :: Normal`
    return ReactiveMP.preprocess_form_constraints(constraints)
end
