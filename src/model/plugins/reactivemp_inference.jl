import GraphPPL: plugin_type, FactorAndVariableNodesPlugin, preprocess_plugin, postprocess_plugin
import GraphPPL: Model, Context, NodeLabel, NodeData, FactorNodeProperties, VariableNodeProperties, NodeCreationOptions, hasextra, getextra, setextra!, getproperties
import GraphPPL: as_variable, is_data, is_random, is_constant, degree
import GraphPPL: variable_nodes, factor_nodes

"""
    ReactiveMPInferenceOptions(; kwargs...)

Creates model inference options object. The list of available options is present below.

### Options

- `limit_stack_depth`: limits the stack depth for computing messages, helps with `StackOverflowError` for some huge models, but reduces the performance of inference backend. Accepts integer as an argument that specifies the maximum number of recursive depth. Lower is better for stack overflow error, but worse for performance.
- `warn`: (optional) flag to suppress warnings. Warnings are not displayed if set to `false`. Defaults to `true`.
- `force_marginal_computation`: (optional) flag to force computation of marginals even when not explicitly requested. Defaults to `false`.

### Advanced options

- `scheduler`: changes the scheduler of reactive streams, see Rocket.jl for more info, defaults to `AsapScheduler`.
- `rulefallback`: specifies a global message update rule fallback for cases when a specific message update rule is not available. Consult `ReactiveMP` documentation for the list of available callbacks.

See also: [`infer`](@ref)
"""
struct ReactiveMPInferenceOptions{S, A, R}
    scheduler::S
    addons::A
    warn::Bool
    force_marginal_computation::Bool
    rulefallback::R
end

ReactiveMPInferenceOptions(scheduler, addons) = ReactiveMPInferenceOptions(scheduler, addons, true, false, nothing)
ReactiveMPInferenceOptions(scheduler, addons, warn) = ReactiveMPInferenceOptions(scheduler, addons, warn, false, nothing)
ReactiveMPInferenceOptions(scheduler, addons, warn, force_marginal_computation) = ReactiveMPInferenceOptions(scheduler, addons, warn, force_marginal_computation, nothing)

setscheduler(options::ReactiveMPInferenceOptions, scheduler) = ReactiveMPInferenceOptions(
    scheduler, options.addons, options.warn, options.force_marginal_computation, options.rulefallback
)
setaddons(options::ReactiveMPInferenceOptions, addons) = ReactiveMPInferenceOptions(
    options.scheduler, addons, options.warn, options.force_marginal_computation, options.rulefallback
)
setwarn(options::ReactiveMPInferenceOptions, warn) = ReactiveMPInferenceOptions(options.scheduler, options.addons, warn, options.force_marginal_computation, options.rulefallback)
setforce_marginal_computation(options::ReactiveMPInferenceOptions, force_marginal_computation) = ReactiveMPInferenceOptions(
    options.scheduler, options.addons, options.warn, force_marginal_computation, options.rulefallback
)
setrulefallback(options::ReactiveMPInferenceOptions, rulefallback) = ReactiveMPInferenceOptions(
    options.scheduler, options.addons, options.warn, options.force_marginal_computation, rulefallback
)

import Base: convert

function Base.convert(::Type{ReactiveMPInferenceOptions}, options::Nothing)
    return convert(ReactiveMPInferenceOptions, (;))
end

function Base.convert(::Type{ReactiveMPInferenceOptions}, options::NamedTuple{keys}) where {keys}
    available_options = (:scheduler, :limit_stack_depth, :addons, :warn, :rulefallback, :force_marginal_computation)

    for key in keys
        key ∈ available_options || error("Unknown model inference options: $(key).")
    end

    warn = haskey(options, :warn) ? options.warn : true
    addons = haskey(options, :addons) ? options.addons : nothing
    rulefallback = haskey(options, :rulefallback) ? options.rulefallback : nothing
    force_marginal_computation = haskey(options, :force_marginal_computation) ? options.force_marginal_computation : false

    if warn && haskey(options, :scheduler) && haskey(options, :limit_stack_depth)
        @warn "Inference options have `scheduler` and `limit_stack_depth` options specified together. Ignoring `limit_stack_depth`. Use `warn = false` option in `ModelInferenceOptions` to suppress this warning."
    end

    scheduler = if haskey(options, :scheduler)
        options[:scheduler]
    elseif haskey(options, :limit_stack_depth)
        LimitStackScheduler(options[:limit_stack_depth]...)
    else
        nothing
    end

    return ReactiveMPInferenceOptions(scheduler, addons, warn, force_marginal_computation, rulefallback)
end

Rocket.getscheduler(options::ReactiveMPInferenceOptions) = something(options.scheduler, AsapScheduler())

import ReactiveMP: getaddons, getrulefallback

ReactiveMP.getaddons(options::ReactiveMPInferenceOptions) = ReactiveMP.getaddons(options, options.addons)
ReactiveMP.getaddons(options::ReactiveMPInferenceOptions, addons::ReactiveMP.AbstractAddon) = (addons,) # ReactiveMP expects addons to be of type tuple
ReactiveMP.getaddons(options::ReactiveMPInferenceOptions, addons::Nothing) = addons                     # Do nothing if addons is `nothing`
ReactiveMP.getaddons(options::ReactiveMPInferenceOptions, addons::Tuple) = addons                       # Do nothing if addons is a `Tuple`
ReactiveMP.getrulefallback(options::ReactiveMPInferenceOptions) = options.rulefallback

# Get the force_marginal_computation setting
getforce_marginal_computation(options::ReactiveMPInferenceOptions) = options.force_marginal_computation

struct ReactiveMPInferencePlugin{Options <: ReactiveMPInferenceOptions}
    options::Options
end

getoptions(plugin::ReactiveMPInferencePlugin) = plugin.options

const ReactiveMPExtraFactorNodeKey = GraphPPL.NodeDataExtraKey{:rmp_factornode, ReactiveMP.AbstractFactorNode}()
const ReactiveMPExtraVariableKey = GraphPPL.NodeDataExtraKey{:rmp_variable, ReactiveMP.AbstractVariable}()
const ReactiveMPExtraDependenciesKey = GraphPPL.NodeDataExtraKey{:dependencies, ReactiveMP.Any}()
const ReactiveMPExtraPipelineKey = GraphPPL.NodeDataExtraKey{:pipeline, ReactiveMP.Any}()

GraphPPL.plugin_type(::ReactiveMPInferencePlugin) = FactorAndVariableNodesPlugin()

function GraphPPL.preprocess_plugin(plugin::ReactiveMPInferencePlugin, model::Model, context::Context, label::NodeLabel, nodedata::NodeData, options::NodeCreationOptions)
    preprocess_plugin(plugin, nodedata, getproperties(nodedata), options)
    return label, nodedata
end

function GraphPPL.preprocess_plugin(plugin::ReactiveMPInferencePlugin, nodedata::NodeData, nodeproperties::VariableNodeProperties, options::NodeCreationOptions)
    return nothing
end

function GraphPPL.preprocess_plugin(plugin::ReactiveMPInferencePlugin, nodedata::NodeData, nodeproperties::FactorNodeProperties, options::NodeCreationOptions)
    if haskey(options, GraphPPL.getkey(ReactiveMPExtraDependenciesKey))
        setextra!(nodedata, ReactiveMPExtraDependenciesKey, options[GraphPPL.getkey(ReactiveMPExtraDependenciesKey)])
    end
    if haskey(options, GraphPPL.getkey(ReactiveMPExtraPipelineKey))
        setextra!(nodedata, ReactiveMPExtraPipelineKey, options[GraphPPL.getkey(ReactiveMPExtraPipelineKey)])
    end
    return nothing
end

function GraphPPL.postprocess_plugin(plugin::ReactiveMPInferencePlugin, model::Model)
    # The variable nodes must be instantiated before the factor nodes
    variable_nodes(model) do label, variable
        properties = getproperties(variable)::VariableNodeProperties

        # Additional check for the model, since `ReactiveMP` does not allow half-edges
        if is_random(properties)
            degree(model, label) !== 0 || error(lazy"Unused random variable has been found $(label).")
            degree(model, label) !== 1 || error(lazy"Half-edge has been found: $(label). To terminate half-edges 'Uninformative' node can be used.")
        end

        set_rmp_variable!(plugin, model, variable, properties)
    end

    # The nodes must be postprocessed after all variables has been instantiated
    factor_nodes(model) do label, factor
        set_rmp_factornode!(plugin, model, factor, getproperties(factor)::FactorNodeProperties)
    end

    # The variable nodes must be activated before the factor nodes
    variable_nodes(model) do _, variable
        activate_rmp_variable!(plugin, model, variable, getproperties(variable)::VariableNodeProperties)
        if hasextra(variable, InitMarExtraKey)
            setmarginal!(getextra(variable, ReactiveMPExtraVariableKey), getextra(variable, InitMarExtraKey))
        end
        if hasextra(variable, InitMsgExtraKey)
            setmessage!(getextra(variable, ReactiveMPExtraVariableKey), getextra(variable, InitMsgExtraKey))
        end
    end

    # The variable nodes must be activated after the variable nodes
    factor_nodes(model) do label, factor
        activate_rmp_factornode!(plugin, model, factor, getproperties(factor)::FactorNodeProperties)
    end
end

function set_rmp_variable!(plugin::ReactiveMPInferencePlugin, model::Model, nodedata::NodeData, nodeproperties::VariableNodeProperties)
    if is_random(nodeproperties)
        return setextra!(nodedata, ReactiveMPExtraVariableKey, randomvar())
    elseif is_data(nodeproperties)
        return setextra!(nodedata, ReactiveMPExtraVariableKey, datavar())
    elseif is_constant(nodeproperties)
        return setextra!(nodedata, ReactiveMPExtraVariableKey, constvar(GraphPPL.value(nodeproperties)))
    else
        error("Unknown `kind` in the node properties `$(nodeproperties)` for variable node `$(nodedata)`. Expected `random`, `constant` or `data`.")
    end
end

function activate_rmp_variable!(plugin::ReactiveMPInferencePlugin, model::Model, nodedata::NodeData, nodeproperties::VariableNodeProperties)
    if is_random(nodeproperties)
        # Fetch "prod-strategy" for messages and marginals. The prod-strategy usually defines the order of messages multiplication (left-to-right)
        # But can use some custom logic for product, e.g. parallel products
        messages_prod_strategy = getextra(nodedata, :messages_prod_strategy, ReactiveMP.FoldLeftProdStrategy())
        marginal_prod_strategy = getextra(nodedata, :marginal_prod_strategy, ReactiveMP.FoldLeftProdStrategy())
        # Fetch "form-constraint" for messages and marginals. The form-constraint usually defines the form of the resulting distribution
        # By default it is `UnspecifiedFormConstraint` which means that the form of the resulting distribution is not specified in advance
        # and follows from the computation, but users may override it with other form constraints, e.g. `PointMassFormConstraint`, which
        # constraints the resulting distribution to be of a point mass form
        messages_form_constraint =
            ReactiveMP.preprocess_form_constraints(
                plugin, model, getextra(nodedata, GraphPPL.VariationalConstraintsMessagesFormConstraintKey, ReactiveMP.UnspecifiedFormConstraint())
            ) + EnsureSupportedFunctionalForm(:μ, GraphPPL.getname(nodeproperties), GraphPPL.index(nodeproperties))
        marginal_form_constraint =
            ReactiveMP.preprocess_form_constraints(
                plugin, model, getextra(nodedata, GraphPPL.VariationalConstraintsMarginalFormConstraintKey, ReactiveMP.UnspecifiedFormConstraint())
            ) + EnsureSupportedFunctionalForm(:q, GraphPPL.getname(nodeproperties), GraphPPL.index(nodeproperties))
        # Fetch "prod-constraint" for messages and marginals. The prod-constraint usually defines the constraints for a single product of messages
        # It can for example preserve a specific parametrization of distribution 
        messages_prod_constraint = getextra(nodedata, :messages_prod_constraint, ReactiveMP.default_prod_constraint(messages_form_constraint))
        marginal_prod_constraint = getextra(nodedata, :marginal_prod_constraint, ReactiveMP.default_prod_constraint(marginal_form_constraint))
        # Fetch "form-check-strategy" for messages and marginals. The form-check-strategy usually defines the strategy for checking the form of the resulting distribution
        # The functional form constraint can be applied either after all products are computed or after each product
        messages_form_check_strategy = getextra(nodedata, :messages_form_check_strategy, ReactiveMP.default_form_check_strategy(messages_form_constraint))
        marginal_form_check_strategy = getextra(nodedata, :marginal_form_check_strategy, ReactiveMP.default_form_check_strategy(marginal_form_constraint))
        # Create the activation options for the random variable which consists of the messages and marginal product functions and a scheduler
        messages_prod_fn = ReactiveMP.messages_prod_fn(messages_prod_strategy, messages_prod_constraint, messages_form_constraint, messages_form_check_strategy)
        marginal_prod_fn = ReactiveMP.marginal_prod_fn(marginal_prod_strategy, marginal_prod_constraint, marginal_form_constraint, marginal_form_check_strategy)
        options = ReactiveMP.RandomVariableActivationOptions(Rocket.getscheduler(getoptions(plugin)), messages_prod_fn, marginal_prod_fn)
        return ReactiveMP.activate!(getextra(nodedata, ReactiveMPExtraVariableKey)::RandomVariable, options)
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
            args = map(arg -> GraphPPL.is_nodelabel(arg) ? getvariable(getvarref(model, arg)) : arg, _args)
        end
        options = ReactiveMP.DataVariableActivationOptions(true, !isnothing(value), transform, args)
        return ReactiveMP.activate!(getextra(nodedata, ReactiveMPExtraVariableKey)::DataVariable, options)
    elseif is_constant(nodeproperties)
        # The constant does not require extra activation
        return nothing
    else
        error("Unknown `kind` in the node properties `$(nodeproperties)` for variable node `$(nodedata)`. Expected `random`, `constant` or `data`.")
    end
end

function set_rmp_factornode!(plugin::ReactiveMPInferencePlugin, model::Model, nodedata::NodeData, nodeproperties::FactorNodeProperties)
    interfaces = map(GraphPPL.neighbors(nodeproperties)) do (_, edge, data)
        return (GraphPPL.getname(edge), getextra(data, ReactiveMPExtraVariableKey))
    end
    factorization = getextra(nodedata, GraphPPL.VariationalConstraintsFactorizationIndicesKey)
    return setextra!(nodedata, ReactiveMPExtraFactorNodeKey, factornode(GraphPPL.fform(nodeproperties), interfaces, factorization))
end

function activate_rmp_factornode!(plugin::ReactiveMPInferencePlugin, model::Model, nodedata::NodeData, nodeproperties::FactorNodeProperties)
    metadata = getextra(nodedata, GraphPPL.MetaExtraKey, nothing)
    dependencies = getextra(nodedata, ReactiveMPExtraDependenciesKey, nothing)
    pipeline = getextra(nodedata, ReactiveMPExtraPipelineKey, nothing)

    scheduler = getscheduler(getoptions(plugin))
    addons = getaddons(getoptions(plugin))
    rulefallback = getrulefallback(getoptions(plugin))

    options = ReactiveMP.FactorNodeActivationOptions(metadata, dependencies, pipeline, addons, scheduler, rulefallback)

    return ReactiveMP.activate!(getextra(nodedata, ReactiveMPExtraFactorNodeKey), options)
end

struct GraphVariableRef
    label::GraphPPL.NodeLabel
    properties::GraphPPL.VariableNodeProperties
    variable::AbstractVariable
end

getlabel(ref::GraphVariableRef) = ref.label
getvariable(ref::GraphVariableRef) = ref.variable
getname(ref::GraphVariableRef) = GraphPPL.getname(getlabel(ref))

GraphPPL.is_data(collection::AbstractArray{GraphVariableRef}) = all(GraphPPL.is_data, collection)

GraphPPL.is_data(ref::GraphVariableRef) = GraphPPL.is_data(ref.properties)
GraphPPL.is_random(ref::GraphVariableRef) = GraphPPL.is_random(ref.properties)
GraphPPL.is_constant(ref::GraphVariableRef) = GraphPPL.is_constant(ref.properties)

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
    return map(v -> getvarref(model, v), GraphPPL.VarDict(GraphPPL.getcontext(model)))
end

getvarref(model::GraphPPL.Model, label::GraphPPL.NodeLabel) = GraphVariableRef(model, label)
getvarref(model::GraphPPL.Model, container::AbstractArray) = map(element -> getvarref(model, element), container)

getvariable(nodedata::GraphPPL.NodeData) = getextra(nodedata, ReactiveMPExtraVariableKey)
getvariable(container::AbstractArray) = map(getvariable, container)

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
        is_constant(getproperties(model[label])::GraphPPL.VariableNodeProperties)
    end
    return map(label -> model[label]::GraphPPL.NodeData, constantlabels)
end

function getfactornodes(model::GraphPPL.Model)
    return map(label -> model[label]::GraphPPL.NodeData, factor_nodes(model))
end

ReactiveMP.israndom(collection::AbstractArray{GraphVariableRef}) = all(ReactiveMP.israndom, collection)
ReactiveMP.isdata(collection::AbstractArray{GraphVariableRef}) = all(ReactiveMP.isdata, collection)
ReactiveMP.isconst(collection::AbstractArray{GraphVariableRef}) = all(ReactiveMP.isconst, collection)

ReactiveMP.israndom(ref::GraphVariableRef) = GraphPPL.is_random(ref.properties)
ReactiveMP.isdata(ref::GraphVariableRef) = GraphPPL.is_data(ref.properties)
ReactiveMP.isconst(ref::GraphVariableRef) = GraphPPL.is_constant(ref.properties)

isanonymous(collection::AbstractArray{GraphVariableRef}) = all(isanonymous, collection)
isanonymous(ref::GraphVariableRef) = GraphPPL.is_anonymous(ref.properties)

ReactiveMP.getmarginal(ref::GraphVariableRef, strategy) = ReactiveMP.getmarginal(ref.variable, strategy)
ReactiveMP.getmarginals(collection::AbstractArray{GraphVariableRef}, strategy) = ReactiveMP.getmarginals(map(ref -> ref.variable, collection), strategy)

ReactiveMP.getprediction(ref::GraphVariableRef) = ReactiveMP.getprediction(ref.variable)
ReactiveMP.getpredictions(collection::AbstractArray{GraphVariableRef}) = ReactiveMP.getpredictions(map(ref -> ref.variable, collection))

ReactiveMP.setmarginal!(ref::GraphVariableRef, marginal) = setmarginal!(ref.variable, marginal)
ReactiveMP.setmarginals!(collection::AbstractArray{GraphVariableRef}, marginal) = ReactiveMP.setmarginals!(map(ref -> ref.variable, collection), marginal)

ReactiveMP.setmessage!(ref::GraphVariableRef, marginal) = setmessage!(ref.variable, marginal)
ReactiveMP.setmessages!(collection::AbstractArray{GraphVariableRef}, marginal) = ReactiveMP.setmessages!(map(ref -> ref.variable, collection), marginal)

# Form constraint preprocessing 

function ReactiveMP.preprocess_form_constraints(backend::ReactiveMPInferencePlugin, model::Model, constraints)
    # It is a simple pass-through for now, but can be extended in the future to preprocess constraints that 
    # are defined in other packages, e.g. in `Distributions` and to support constraints, such as `q(x) :: Normal`
    return ReactiveMP.preprocess_form_constraints(constraints)
end
