
export FactorGraphModel
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

- `pipeline`: changes the default pipeline for each factor node in the graph
- `global_reactive_scheduler`: changes the scheduler of reactive streams, see Rocket.jl for more info, defaults to no scheduler

See also: [`infer`](@ref)
"""
struct InferenceOptions{P, S, A}
    pipeline  :: P
    scheduler :: S
    addons    :: A
    warn      :: Bool
end

InferenceOptions(pipeline, scheduler, addons) = InferenceOptions(pipeline, scheduler, addons, true)

UnspecifiedInferenceOptions() = convert(InferenceOptions, (;))

setpipeline(options::InferenceOptions, pipeline) = InferenceOptions(pipeline, options.scheduler, options.addons, options.warn)
setscheduler(options::InferenceOptions, scheduler) = InferenceOptions(options.pipeline, scheduler, options.addons, options.warn)
setaddons(options::InferenceOptions, addons) = InferenceOptions(options.pipeline, options.scheduler, addons, options.warn)
setwarn(options::InferenceOptions, warn) = InferenceOptions(options.pipeline, options.scheduler, options.addons, warn)

import Base: convert

function Base.convert(::Type{InferenceOptions}, options::Nothing)
    return UnspecifiedInferenceOptions()
end

function Base.convert(::Type{InferenceOptions}, options::NamedTuple{keys}) where {keys}
    available_options = (:pipeline, :scheduler, :limit_stack_depth, :addons, :warn)

    for key in keys
        key ∈ available_options || error("Unknown model inference options: $(key).")
    end

    pipeline  = nothing
    scheduler = nothing
    addons    = nothing
    warn      = true

    if haskey(options, :warn)
        warn = options[:warn]
    end

    if haskey(options, :pipeline)
        pipeline = options[:pipeline]
    end

    if warn && haskey(options, :scheduler) && haskey(options, :limit_stack_depth)
        @warn "Model options have `scheduler` and `limit_stack_depth` options specified together. Ignoring `limit_stack_depth`. Use `warn = false` option in `ModelInferenceOptions` to suppress this warning."
    end

    if haskey(options, :scheduler)
        scheduler = options[:scheduler]
    elseif haskey(options, :limit_stack_depth)
        scheduler = LimitStackScheduler(options[:limit_stack_depth]...)
    end

    if haskey(options, :addons)
        addons = options[:addons]
    end

    return InferenceOptions(pipeline, scheduler, addons, warn)
end

const DefaultModelInferenceOptions = UnspecifiedInferenceOptions()

Rocket.getscheduler(options::InferenceOptions) = something(options.scheduler, AsapScheduler())

ReactiveMP.get_pipeline_stages(options::InferenceOptions) = something(options.pipeline, EmptyPipelineStage())

ReactiveMP.getaddons(options::InferenceOptions) = ReactiveMP.getaddons(options, options.addons)
ReactiveMP.getaddons(options::InferenceOptions, addons::ReactiveMP.AbstractAddon) = (addons,) # ReactiveMP expects addons to be of type tuple
ReactiveMP.getaddons(options::InferenceOptions, addons::Nothing) = addons                      # Do nothing if addons is `nothing`
ReactiveMP.getaddons(options::InferenceOptions, addons::Tuple) = addons                        # Do nothing if addons is a `Tuple`

struct FactorGraphModel{M}
    model::M
end

getmodel(model::FactorGraphModel) = model.model

getvardict(model::FactorGraphModel) = getvardict(getmodel(model))

function getvardict(model::GraphPPL.Model)
    ctx = GraphPPL.getcontext(model)
    vardict = Dict{Symbol, Tuple{NodeData, VariableNodeProperties}}()
    # TODO very ugly code just to make things running
    return map(merge(ctx.individual_variables, ctx.vector_variables)) do value
        nodedata = model[value]
        if nodedata isa GraphPPL.NodeData
            return (nodedata, GraphPPL.getproperties(nodedata))
        else
            return map(v -> (v, GraphPPL.getproperties(v)), nodedata)
        end
    end
    return vardict
end

# TODO very ugly code just to make things running
import GraphPPL: NodeData, VariableNodeProperties

# TODO very ugly code just to make things running
ReactiveMP.allows_missings(::Vector{Tuple{NodeData, VariableNodeProperties}}) = false
ReactiveMP.allows_missings(::Tuple{NodeData, VariableNodeProperties}) = false

# TODO very ugly code just to make things running
ReactiveMP.israndom(t::Vector{Tuple{NodeData, VariableNodeProperties}}) = all(ReactiveMP.israndom, t)
ReactiveMP.isdata(t::Vector{Tuple{NodeData, VariableNodeProperties}}) = all(ReactiveMP.isdata, t)
ReactiveMP.isconst(t::Vector{Tuple{NodeData, VariableNodeProperties}}) = all(ReactiveMP.isconst, t)

# TODO very ugly code just to make things running
ReactiveMP.israndom(t::Tuple{NodeData, VariableNodeProperties}) = GraphPPL.is_random(t[2])
ReactiveMP.isdata(t::Tuple{NodeData, VariableNodeProperties}) = GraphPPL.is_data(t[2])
ReactiveMP.isconst(t::Tuple{NodeData, VariableNodeProperties}) = GraphPPL.is_constant(t[2])

function ReactiveMP.getmarginal(t::Tuple{NodeData, VariableNodeProperties}, strategy)
    return ReactiveMP.getmarginal(getextra(t[1], :rmp_properties), strategy)
end

function ReactiveMP.getmarginals(t::Vector{Tuple{NodeData, VariableNodeProperties}}, strategy)
    tt = map(t) do _t
        getextra(_t[1], :rmp_properties)
    end
    return ReactiveMP.getmarginals(tt, strategy)
end

function ReactiveMP.update!(t::Vector{Tuple{GraphPPL.NodeData, GraphPPL.VariableNodeProperties}}, something)
    rmpprops = map(i -> getextra(i[1], :rmp_properties), t)
    return ReactiveMP.update!(rmpprops, something)
end

# getmodel(model::FactorGraphModel)     = model.model
# getvariables(model::FactorGraphModel) = getvariables(model, getmodel(model))
# 
# # If the underlying model is a `GraphPPL.Model` then we simply return the underlying context
# getvariables(::FactorGraphModel, model::GraphPPL.Model) = GraphPPL.getcontext(model)
# 
# getconstraints(model::FactorGraphModel) = model.constraints
# getmeta(model::FactorGraphModel)        = model.meta
# getoptions(model::FactorGraphModel)     = model.options
# getnodes(model::FactorGraphModel)       = model.nodes
# 
# import ReactiveMP: getrandom, getconstant, getdata, getvardict
# 
# ReactiveMP.getrandom(model::FactorGraphModel)   = getrandom(getvariables(model))
# ReactiveMP.getconstant(model::FactorGraphModel) = getconstant(getvariables(model))
# ReactiveMP.getdata(model::FactorGraphModel)     = getdata(getvariables(model))
# ReactiveMP.getvardict(model::FactorGraphModel)  = getvardict(getvariables(model))
# 
# function Base.getindex(model::FactorGraphModel, symbol::Symbol)
#     return getindex(getmodel(model), getindex(getvariables(model), symbol))
# end
# 
# function Base.haskey(model::FactorGraphModel, symbol::Symbol)
#     return haskey(getvariables(model), symbol)
# end

# Base.broadcastable(model::FactorGraphModel) = (model,)

import ReactiveMP: hasrandomvar, hasdatavar, hasconstvar

# ReactiveMP.hasrandomvar(model::FactorGraphModel, symbol::Symbol) = hasrandomvar(getvariables(model), symbol)
# ReactiveMP.hasdatavar(model::FactorGraphModel, symbol::Symbol)   = hasdatavar(getvariables(model), symbol)
# ReactiveMP.hasconstvar(model::FactorGraphModel, symbol::Symbol)  = hasconstvar(getvariables(model), symbol)

##  We extend integrate `ReactiveMP` and `GraphPPL` functionalities here

import GraphPPL: plugin_type, FactorAndVariableNodesPlugin, preprocess_plugin, postprocess_plugin
import GraphPPL: Model, Context, NodeLabel, NodeData, FactorNodeProperties, NodeCreationOptions, hasextra, getextra, setextra!, getproperties
import GraphPPL: as_variable, is_data, is_random, is_constant, degree
import GraphPPL: variable_nodes, factor_nodes

"""
- `options`: An instance of `InferenceOptions`
"""
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
    preprocess_reactivemp_plugin!(plugin, model, context, nodedata, getproperties(nodedata), options)
    return label, nodedata
end

function preprocess_reactivemp_plugin!(
    plugin::ReactiveMPIntegrationPlugin, model::Model, context::Context, nodedata::NodeData, nodeproperties::FactorNodeProperties, options::NodeCreationOptions
)
    interfaces = map(options[:interfaces]) do interface
        inodedata = model[interface]::NodeData
        iproperties = getproperties(inodedata)::VariableNodeProperties
        if !hasextra(inodedata, :rmp_properties)
            preprocess_reactivemp_plugin!(plugin, model, context, inodedata, iproperties, NodeCreationOptions())
        end
        return (inodedata, iproperties)
    end

    setextra!(nodedata, :rmp_properties, ReactiveMP.FactorNodeProperties(interfaces))

    return nothing
end

function preprocess_reactivemp_plugin!(
    ::ReactiveMPIntegrationPlugin, model::Model, context::Context, nodedata::NodeData, nodeproperties::VariableNodeProperties, options::NodeCreationOptions
)
    if is_random(nodeproperties)
        setextra!(nodedata, :rmp_properties, RandomVariable()) # TODO: bvdmitri, use functional form constraints
    elseif is_data(nodeproperties)
        setextra!(nodedata, :rmp_properties, DataVariable())
    elseif is_constant(nodeproperties)
        setextra!(nodedata, :rmp_properties, ConstVariable(GraphPPL.value(nodeproperties)))
    else
        error("Unknown `kind` in the node properties `$(nodeproperties)` for variable node `$(nodedata)`. Expected `random`, `constant` or `data`.")
    end
    return nothing
end

function GraphPPL.postprocess_plugin(plugin::ReactiveMPIntegrationPlugin, model::Model)
    # The variable nodes must be postprocessed before the factor nodes
    variable_nodes(model) do label, variable
        properties = getproperties(variable)::VariableNodeProperties

        # Additional check for the model, since `ReactiveMP` does not allow half-edges
        if is_random(properties)
            degree(model, label) !== 0 || error(lazy"Unused random variable has been found $(label).")
            degree(model, label) !== 1 || error(lazy"Half-edge has been found: $(label). To terminate half-edges 'Uninformative' node can be used.")
        end

        postprocess_reactivemp_node(plugin, model, variable, properties)
    end

    # The nodes must be postprocessed after all variables has been instantiated
    factor_nodes(model) do label, factor
        properties = getproperties(factor)::FactorNodeProperties

        postprocess_reactivemp_node(plugin, model, factor, properties)
    end
end

function postprocess_reactivemp_node(plugin::ReactiveMPIntegrationPlugin, model::Model, nodedata::NodeData, nodeproperties::VariableNodeProperties)
    if is_random(nodeproperties)
        ReactiveMP.activate!(getextra(nodedata, :rmp_properties)::RandomVariable, ReactiveMP.RandomVariableActivationOptions(Rocket.getscheduler(getoptions(plugin))))
    elseif is_data(nodeproperties)
        ReactiveMP.activate!(getextra(nodedata, :rmp_properties)::DataVariable, ReactiveMP.DataVariableActivationOptions(false))
    elseif is_constant(nodeproperties)
        # Properties for constant labels do not require extra activation
        return nothing
    else
        error("Unknown `kind` in the node properties `$(nodeproperties)` for variable node `$(nodedata)`. Expected `random`, `constant` or `data`.")
    end
    return nothing
end

function postprocess_reactivemp_node(plugin::ReactiveMPIntegrationPlugin, model::Model, nodedata::NodeData, nodeproperties::FactorNodeProperties)

    # TODO: bvdmitri, Wouter is working on a faster version of this
    factorization = Tuple.(getextra(nodedata, :factorization_constraint_indices))
    metadata = hasextra(nodedata, :meta) ? getextra(nodedata, :meta) : nothing
    pipeline = hasextra(nodedata, :pipeline) ? getextra(nodedata, :pipeline) : nothing
    scheduler = getscheduler(getoptions(plugin))
    addons = getaddons(getoptions(plugin))
    options = ReactiveMP.FactorNodeActivationOptions(GraphPPL.fform(nodeproperties), factorization, metadata, pipeline, addons, scheduler)

    ReactiveMP.activate!(getextra(nodedata, :rmp_properties)::ReactiveMP.FactorNodeProperties, options)
    return nothing
end

## ReactiveMP <-> GraphPPL connections, techically a piracy, but that is the purpose of the plugin
function Base.convert(::Type{ReactiveMP.AbstractVariable}, spec::Tuple{NodeData, VariableNodeProperties})
    nodedata, nodeproperties = spec
    if is_random(nodeproperties)
        return getextra(nodedata, :rmp_properties)::RandomVariable
    elseif is_data(nodeproperties)
        return getextra(nodedata, :rmp_properties)::DataVariable
    elseif is_constant(nodeproperties)
        return getextra(nodedata, :rmp_properties)::ConstVariable
    else
        error("Unknown `kind` in the node properties `$(nodeproperties)` for variable node `$(nodedata)`. Expected `random`, `constant` or `data`.")
    end
end