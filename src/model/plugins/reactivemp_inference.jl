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

### Advanced options

- `scheduler`: changes the scheduler of reactive streams, see Rocket.jl for more info, defaults to no scheduler

See also: [`infer`](@ref)
"""
struct ReactiveMPInferenceOptions{S, A}
    scheduler :: S
    addons    :: A
    warn      :: Bool
end

ReactiveMPInferenceOptions(scheduler, addons) = ReactiveMPInferenceOptions(scheduler, addons, true)

setscheduler(options::ReactiveMPInferenceOptions, scheduler) = ReactiveMPInferenceOptions(scheduler, options.addons, options.warn)
setaddons(options::ReactiveMPInferenceOptions, addons) = ReactiveMPInferenceOptions(options.scheduler, addons, options.warn)
setwarn(options::ReactiveMPInferenceOptions, warn) = ReactiveMPInferenceOptions(options.scheduler, options.addons, warn)

import Base: convert

function Base.convert(::Type{ReactiveMPInferenceOptions}, options::Nothing)
    return convert(ReactiveMPInferenceOptions, (;))
end

function Base.convert(::Type{ReactiveMPInferenceOptions}, options::NamedTuple{keys}) where {keys}
    available_options = (:scheduler, :limit_stack_depth, :addons, :warn)

    for key in keys
        key ∈ available_options || error("Unknown model inference options: $(key).")
    end

    warn = haskey(options, :warn) ? options.warn : true
    addons = haskey(options, :addons) ? options.addons : nothing

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

    return ReactiveMPInferenceOptions(scheduler, addons, warn)
end

Rocket.getscheduler(options::ReactiveMPInferenceOptions) = something(options.scheduler, AsapScheduler())

ReactiveMP.getaddons(options::ReactiveMPInferenceOptions) = ReactiveMP.getaddons(options, options.addons)
ReactiveMP.getaddons(options::ReactiveMPInferenceOptions, addons::ReactiveMP.AbstractAddon) = (addons,) # ReactiveMP expects addons to be of type tuple
ReactiveMP.getaddons(options::ReactiveMPInferenceOptions, addons::Nothing) = addons                     # Do nothing if addons is `nothing`
ReactiveMP.getaddons(options::ReactiveMPInferenceOptions, addons::Tuple) = addons                       # Do nothing if addons is a `Tuple`

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
        messages_form_constraint = getextra(nodedata, GraphPPL.VariationalConstraintsMessagesFormConstraintKey, ReactiveMP.UnspecifiedFormConstraint())
        marginal_form_constraint = getextra(nodedata, GraphPPL.VariationalConstraintsMarginalFormConstraintKey, ReactiveMP.UnspecifiedFormConstraint())
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
        # TODO: bvdmitri use allow_missings
        return ReactiveMP.activate!(getextra(nodedata, ReactiveMPExtraVariableKey)::DataVariable, ReactiveMP.DataVariableActivationOptions(false))
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
    options = ReactiveMP.FactorNodeActivationOptions(metadata, dependencies, pipeline, addons, scheduler)

    return ReactiveMP.activate!(getextra(nodedata, ReactiveMPExtraFactorNodeKey), options)
end

struct GraphVariableRef
    label::GraphPPL.NodeLabel
    properties::GraphPPL.VariableNodeProperties
    variable::AbstractVariable
end

function GraphVariableRef(model::GraphPPL.Model, label::GraphPPL.NodeLabel)
    nodedata = model[label]::GraphPPL.NodeData
    properties = getproperties(nodedata)::GraphPPL.VariableNodeProperties
    variable = getextra(nodedata, :rmp_variable)::AbstractVariable
    return GraphVariableRef(label, properties, variable)
end

function getvardict(model::GraphPPL.Model)
    return map(v -> getvarref(model, v), GraphPPL.VarDict(GraphPPL.getcontext(model)))
end

getvarref(model::GraphPPL.Model, label::GraphPPL.NodeLabel) = GraphVariableRef(model, label)
getvarref(model::GraphPPL.Model, container::AbstractArray) = map(element -> getvarref(model, element), container)

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

ReactiveMP.allows_missings(::AbstractArray{GraphVariableRef}) = false
ReactiveMP.allows_missings(::GraphVariableRef) = false

ReactiveMP.israndom(collection::AbstractArray{GraphVariableRef}) = all(ReactiveMP.israndom, collection)
ReactiveMP.isdata(collection::AbstractArray{GraphVariableRef}) = all(ReactiveMP.isdata, collection)
ReactiveMP.isconst(collection::AbstractArray{GraphVariableRef}) = all(ReactiveMP.isconst, collection)

ReactiveMP.israndom(ref::GraphVariableRef) = GraphPPL.is_random(ref.properties)
ReactiveMP.isdata(ref::GraphVariableRef) = GraphPPL.is_data(ref.properties)
ReactiveMP.isconst(ref::GraphVariableRef) = GraphPPL.is_constant(ref.properties)

ReactiveMP.getmarginal(ref::GraphVariableRef, strategy) = ReactiveMP.getmarginal(ref.variable, strategy)
ReactiveMP.getmarginals(collection::AbstractArray{GraphVariableRef}, strategy) = ReactiveMP.getmarginals(map(ref -> ref.variable, collection), strategy)

ReactiveMP.setmarginal!(ref::GraphVariableRef, marginal) = setmarginal!(ref.variable, marginal)
ReactiveMP.setmarginals!(collection::AbstractArray{GraphVariableRef}, marginal) = ReactiveMP.setmarginals!(map(ref -> ref.variable, collection), marginal)

ReactiveMP.setmessage!(ref::GraphVariableRef, marginal) = setmessage!(ref.variable, marginal)
ReactiveMP.setmessages!(collection::AbstractArray{GraphVariableRef}, marginal) = ReactiveMP.setmessages!(map(ref -> ref.variable, collection), marginal)

function ReactiveMP.update!(ref::GraphVariableRef, something)
    # TODO: cache the result of the `getextra` call in the `inference` procedure
    return ReactiveMP.update!(ref.variable, something)
end

function ReactiveMP.update!(collection::AbstractArray{<:GraphVariableRef}, something)
    # TODO: cache the result of the `getextra` call in the `inference` procedure
    return ReactiveMP.update!(map(ref -> ref.variable, collection), something)
end