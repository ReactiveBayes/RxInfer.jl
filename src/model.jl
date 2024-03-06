
export ProbabilisticModel
export getoptions, getconstraints, getmeta
export getnodes, getvariables, getrandom, getconstant, getdata

import Base: push!, show, getindex, haskey, firstindex, lastindex
import ReactiveMP: get_pipeline_stages, getaddons, AbstractFactorNode
import Rocket: getscheduler

# Model Inference Options

"""
    ModelInferenceOptions(; kwargs...)

Creates model inference options object. The list of available options is present below.

### Options

- `limit_stack_depth`: limits the stack depth for computing messages, helps with `StackOverflowError` for some huge models, but reduces the performance of inference backend. Accepts integer as an argument that specifies the maximum number of recursive depth. Lower is better for stack overflow error, but worse for performance.
- `warn`: (optional) flag to suppress warnings. Warnings are not displayed if set to `false`. Defaults to `true`.

### Advanced options

- `scheduler`: changes the scheduler of reactive streams, see Rocket.jl for more info, defaults to no scheduler

See also: [`infer`](@ref)
"""
struct InferenceOptions{S, A}
    scheduler :: S
    addons    :: A
    warn      :: Bool
end

InferenceOptions(scheduler, addons) = InferenceOptions(scheduler, addons, true)

UnspecifiedInferenceOptions() = convert(InferenceOptions, (;))

setscheduler(options::InferenceOptions, scheduler) = InferenceOptions(scheduler, options.addons, options.warn)
setaddons(options::InferenceOptions, addons) = InferenceOptions(options.scheduler, addons, options.warn)
setwarn(options::InferenceOptions, warn) = InferenceOptions(options.scheduler, options.addons, warn)

import Base: convert

function Base.convert(::Type{InferenceOptions}, options::Nothing)
    return UnspecifiedInferenceOptions()
end

function Base.convert(::Type{InferenceOptions}, options::NamedTuple{keys}) where {keys}
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

    return InferenceOptions(scheduler, addons, warn)
end

const DefaultModelInferenceOptions = UnspecifiedInferenceOptions()

Rocket.getscheduler(options::InferenceOptions) = something(options.scheduler, AsapScheduler())

ReactiveMP.getaddons(options::InferenceOptions) = ReactiveMP.getaddons(options, options.addons)
ReactiveMP.getaddons(options::InferenceOptions, addons::ReactiveMP.AbstractAddon) = (addons,) # ReactiveMP expects addons to be of type tuple
ReactiveMP.getaddons(options::InferenceOptions, addons::Nothing) = addons                     # Do nothing if addons is `nothing`
ReactiveMP.getaddons(options::InferenceOptions, addons::Tuple) = addons                       # Do nothing if addons is a `Tuple`

struct ProbabilisticModel{M}
    model::M
end

getmodel(model::ProbabilisticModel) = model.model

getvardict(model::ProbabilisticModel) = getvardict(getmodel(model))
getrandomvars(model::ProbabilisticModel) = getrandomvars(getmodel(model))
getfactornodes(model::ProbabilisticModel) = getfactornodes(getmodel(model))

import GraphPPL: plugin_type, FactorAndVariableNodesPlugin, preprocess_plugin, postprocess_plugin
import GraphPPL: Model, Context, NodeLabel, NodeData, FactorNodeProperties, VariableNodeProperties, NodeCreationOptions, hasextra, getextra, setextra!, getproperties
import GraphPPL: as_variable, is_data, is_random, is_constant, degree
import GraphPPL: variable_nodes, factor_nodes

struct ReactiveMPIntegrationPlugin{T}
    options::T

    function ReactiveMPIntegrationPlugin(options::T) where {T <: InferenceOptions}
        return new{T}(options)
    end
end

ReactiveMPIntegrationPlugin(options) = ReactiveMPIntegrationPlugin(convert(InferenceOptions, options))

getoptions(plugin::ReactiveMPIntegrationPlugin) = plugin.options

GraphPPL.plugin_type(::ReactiveMPIntegrationPlugin) = FactorAndVariableNodesPlugin()

function GraphPPL.preprocess_plugin(plugin::ReactiveMPIntegrationPlugin, model::Model, context::Context, label::NodeLabel, nodedata::NodeData, options::NodeCreationOptions)
    return label, nodedata
end

function GraphPPL.postprocess_plugin(plugin::ReactiveMPIntegrationPlugin, model::Model)
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

function set_rmp_variable!(plugin::ReactiveMPIntegrationPlugin, model::Model, nodedata::NodeData, nodeproperties::VariableNodeProperties)
    if is_random(nodeproperties)
        return setextra!(nodedata, :rmp_variable, randomvar())
    elseif is_data(nodeproperties)
        return setextra!(nodedata, :rmp_variable, datavar())
    elseif is_constant(nodeproperties)
        return setextra!(nodedata, :rmp_variable, constvar(GraphPPL.value(nodeproperties)))
    else
        error("Unknown `kind` in the node properties `$(nodeproperties)` for variable node `$(nodedata)`. Expected `random`, `constant` or `data`.")
    end
end

function activate_rmp_variable!(plugin::ReactiveMPIntegrationPlugin, model::Model, nodedata::NodeData, nodeproperties::VariableNodeProperties)
    if is_random(nodeproperties)
        # TODO: bvdmitri, use functional form constraints
        return ReactiveMP.activate!(getextra(nodedata, :rmp_variable)::RandomVariable, ReactiveMP.RandomVariableActivationOptions(Rocket.getscheduler(getoptions(plugin))))
    elseif is_data(nodeproperties)
        # TODO: bvdmitri use allow_missings
        return ReactiveMP.activate!(getextra(nodedata, :rmp_variable)::DataVariable, ReactiveMP.DataVariableActivationOptions(false))
    elseif is_constant(nodeproperties)
        # The constant does not require extra activation
        return nothing
    else
        error("Unknown `kind` in the node properties `$(nodeproperties)` for variable node `$(nodedata)`. Expected `random`, `constant` or `data`.")
    end
end

function set_rmp_factornode!(plugin::ReactiveMPIntegrationPlugin, model::Model, nodedata::NodeData, nodeproperties::FactorNodeProperties)
    interfaces = map(GraphPPL.neighbors(nodeproperties)) do (label, edge)
        return (GraphPPL.getname(edge), getextra(model[label], :rmp_variable))
    end
    return setextra!(nodedata, :rmp_factornode, factornode(GraphPPL.fform(nodeproperties), interfaces))
end

function activate_rmp_factornode!(plugin::ReactiveMPIntegrationPlugin, model::Model, nodedata::NodeData, nodeproperties::FactorNodeProperties)
    factorization = getextra(nodedata, :factorization_constraint_indices)
    metadata = hasextra(nodedata, :meta) ? getextra(nodedata, :meta) : nothing
    dependencies = hasextra(nodedata, :dependencies) ? getextra(nodedata, :dependencies) : nothing
    pipeline = hasextra(nodedata, :pipeline) ? getextra(nodedata, :pipeline) : nothing

    scheduler = getscheduler(getoptions(plugin))
    addons = getaddons(getoptions(plugin))
    options = ReactiveMP.FactorNodeActivationOptions(factorization, metadata, dependencies, pipeline, addons, scheduler)

    return ReactiveMP.activate!(getextra(nodedata, :rmp_factornode), options)
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
    # TODO replace with filter predicate
    return Iterators.filter(GraphPPL.labels(model)) do label
        error(1)
    end
end

function getfactornodes(model::GraphPPL.Model)
    # TODO replace with filter predicate
    return map(factor_nodes(model)) do label
        return getextra(model[label], :rmp_properties)
    end
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

function ReactiveMP.update!(collection::Vector{GraphVariableRef}, something)
    # TODO: cache the result of the `getextra` call in the `inference` procedure
    return ReactiveMP.update!(map(ref -> ref.variable, collection), something)
end