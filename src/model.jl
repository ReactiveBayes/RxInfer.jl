export AbstractModelSpecification
export ModelOptions, model_options
export FactorGraphModel, Model
export AutoVar
export getoptions, getconstraints, getmeta, getstats
export getnodes, getrandom, getconstant, getdata

import Base: show, getindex, haskey, firstindex, lastindex
import ReactiveMP: AbstractFactorNode

# Abstract model specification

abstract type AbstractModelSpecification end

function create_model end

function model_name end

function source_code end

# Model Generator

"""
    ModelGenerator

`ModelGenerator` is a special object that is used in the `inference` function to lazily create model later on given `constraints`, `meta` and `options`.

See also: [`inference`](@ref)
"""
struct ModelGenerator{T, A, K, C, M, O}
    args        :: A
    kwargs      :: K
    constraints :: C
    meta        :: M
    options     :: O

    ModelGenerator(::Type{T}, args::A, kwargs::K, constraints::C, meta::M, options::O) where {T, A, K, C, M, O} =
        new{T, A, K, C, M, O}(args, kwargs, constraints, meta, options)
end

# `ModelGenerator{T, A, K, Nothing, Nothing, Nothing}` is returned from the `Model` function
function create_model(
    generator::ModelGenerator{T, A, K, Nothing, Nothing, Nothing},
    constraints,
    meta,
    options
) where {T, A, K}
    return create_model(T, constraints, meta, options, generator.args...; generator.kwargs...)
end

function create_model(generator::ModelGenerator{T}) where {T}
    return create_model(
        T,
        generator.constraints,
        generator.meta,
        generator.options,
        generator.args...;
        generator.kwargs...
    )
end

# Model Options

struct ModelOptions{P, F, S}
    pipeline                  :: P
    default_factorisation     :: F
    global_reactive_scheduler :: S
end

"""
    model_options(options...)

Creates model options object. The list of available options is present below:

### Options

- `default_factorisation`: specifies default factorisation for all factor nodes, e.g. `MeanField()` or `FullFactorisation`. **Note**: this setting is not compatible with `@constraints`
- `limit_stack_depth`: limits the stack depth for computing messages, helps with `StackOverflowError` for some huge models, but reduces the performance of inference backend. Accepts integer as an argument that specifies the maximum number of recursive depth. Lower is better for stack overflow error, but worse for performance.

### Advanced options

- `pipeline`: changes the default pipeline for each factor node in the graph
- `global_reactive_scheduler`: changes the scheduler of reactive streams, see Rocket.jl for more info, defaults to no scheduler

See also: [`inference`](@ref)
"""
function model_options end

model_options(; kwargs...) = model_options(kwargs)

model_options(pairs::Base.Iterators.Pairs) = model_options(NamedTuple(pairs))

available_option_names(::Type{<:ModelOptions}) = (
    :pipeline,
    :default_factorisation,
    :global_reactive_scheduler,
    :limit_stack_depth
)

__as_named_tuple(nt::NamedTuple, arg1::NamedTuple{T, Tuple{Nothing}}, tail...) where {T} = __as_named_tuple(nt, tail...)
__as_named_tuple(nt::NamedTuple, arg1::NamedTuple, tail...)                              = __as_named_tuple(merge(nt, arg1), tail...)

__as_named_tuple(nt::NamedTuple) = nt

as_named_tuple(options::ModelOptions) = __as_named_tuple((;),
    (pipeline = options.pipeline,),
    (default_factorisation = options.default_factorisation,),
    (global_reactive_scheduler = options.global_reactive_scheduler,)
)

function model_options(options::NamedTuple)
    pipeline                  = nothing
    default_factorisation     = nothing
    global_reactive_scheduler = nothing

    if haskey(options, :pipeline)
        pipeline = options[:pipeline]
    end

    if haskey(options, :default_factorisation)
        default_factorisation = options[:default_factorisation]
    end

    if haskey(options, :global_reactive_scheduler) && haskey(options, :limit_stack_depth)
        @warn "Model options have `global_reactive_scheduler` and `limit_stack_depth` options specified together. Ignoring `limit_stack_depth`."
    end

    if haskey(options, :global_reactive_scheduler)
        global_reactive_scheduler = options[:global_reactive_scheduler]
    elseif haskey(options, :limit_stack_depth)
        global_reactive_scheduler = LimitStackScheduler(options[:limit_stack_depth]...)
    end

    for key::Symbol in
        setdiff(union(available_option_names(ModelOptions), fields(options)), available_option_names(ModelOptions))
        @warn "Unknown option key: $key = $(options[key])"
    end

    return ModelOptions(
        pipeline,
        default_factorisation,
        global_reactive_scheduler
    )
end

global_reactive_scheduler(options::ModelOptions) = something(options.global_reactive_scheduler, AsapScheduler())
get_pipeline_stages(options::ModelOptions)       = something(options.pipeline, EmptyPipelineStage())
default_factorisation(options::ModelOptions)     = something(options.default_factorisation, UnspecifiedConstraints())

Base.merge(nt::NamedTuple, options::ModelOptions) = model_options(merge(nt, as_named_tuple(options)))

# Model

Model(::Type{T}, args...; kwargs...) where {T <: AbstractModelSpecification} =
    ModelGenerator(T, args, kwargs, nothing, nothing, nothing)

struct FactorGraphModelStats
    node_ids::Set{Symbol}

    FactorGraphModelStats() = new(Set{Symbol}())
end

add!(stats::FactorGraphModelStats, node::AbstractFactorNode) =
    push!(stats.node_ids, as_node_symbol(functionalform(node)))

hasnodeid(stats::FactorGraphModelStats, nodeid::Symbol) = nodeid ∈ stats.node_ids

struct FactorGraphModel{C, M, O}
    constraints :: C
    meta        :: M
    options     :: O
    nodes       :: Vector{AbstractFactorNode}
    random      :: Vector{RandomVariable}
    constant    :: Vector{ConstVariable}
    data        :: Vector{DataVariable}
    vardict     :: Dict{Symbol, Any}
    stats       :: FactorGraphModelStats
end

Base.show(io::IO, ::Type{<:FactorGraphModel}) = print(io, "FactorGraphModel")
Base.show(io::IO, model::FactorGraphModel)    = print(io, "FactorGraphModel()")

FactorGraphModel() = FactorGraphModel(DefaultConstraints, DefaultMeta, model_options())

FactorGraphModel(constraints::Union{UnspecifiedConstraints, ConstraintsSpecification}) = FactorGraphModel(constraints, DefaultMeta, model_options())
FactorGraphModel(meta::Union{UnspecifiedMeta, MetaSpecification})                      = FactorGraphModel(DefaultConstraints, meta, model_options())
FactorGraphModel(options::NamedTuple)                                                  = FactorGraphModel(DefaultConstraints, DefaultMeta, model_options(options))

FactorGraphModel(constraints::Union{UnspecifiedConstraints, ConstraintsSpecification}, options::NamedTuple) = FactorGraphModel(constraints, DefaultMeta, model_options(options))
FactorGraphModel(meta::Union{UnspecifiedMeta, MetaSpecification}, options::NamedTuple)                      = FactorGraphModel(DefaultConstraints, meta, model_options(options))

FactorGraphModel(constraints::Union{UnspecifiedConstraints, ConstraintsSpecification}, meta::Union{UnspecifiedMeta, MetaSpecification})                      = FactorGraphModel(constraints, meta, model_options())
FactorGraphModel(constraints::Union{UnspecifiedConstraints, ConstraintsSpecification}, meta::Union{UnspecifiedMeta, MetaSpecification}, options::NamedTuple) = FactorGraphModel(constraints, meta, model_options(options))

function FactorGraphModel(
    constraints::C,
    meta::M,
    options::O
) where {
    C <: Union{UnspecifiedConstraints, ConstraintsSpecification},
    M <: Union{UnspecifiedMeta, MetaSpecification},
    O <: ModelOptions
}
    return FactorGraphModel{C, M, O}(
        constraints,
        meta,
        options,
        Vector{FactorNode}(),
        Vector{RandomVariable}(),
        Vector{ConstVariable}(),
        Vector{DataVariable}(),
        Dict{Symbol, Any}(),
        FactorGraphModelStats()
    )
end

getconstraints(model::FactorGraphModel) = model.constraints
getmeta(model::FactorGraphModel)        = model.meta
getoptions(model::FactorGraphModel)     = model.options
getnodes(model::FactorGraphModel)       = model.nodes
getrandom(model::FactorGraphModel)      = model.random
getconstant(model::FactorGraphModel)    = model.constant
getdata(model::FactorGraphModel)        = model.data
getvardict(model::FactorGraphModel)     = model.vardict
getstats(model::FactorGraphModel)       = model.stats

function Base.getindex(model::FactorGraphModel, symbol::Symbol)
    vardict = getvardict(model)
    if !haskey(vardict, symbol)
        error("Model has no variable/variables named $(symbol).")
    end
    return getindex(getvardict(model), symbol)
end

function Base.haskey(model::FactorGraphModel, symbol::Symbol)
    return haskey(getvardict(model), symbol)
end

Base.broadcastable(model::FactorGraphModel) = Ref(model)

hasrandomvar(model::FactorGraphModel, symbol::Symbol) = haskey(model, symbol) ? israndom(getindex(model, symbol)) : false
hasdatavar(model::FactorGraphModel, symbol::Symbol)   = haskey(model, symbol) ? isdata(getindex(model, symbol)) : false
hasconstvar(model::FactorGraphModel, symbol::Symbol)  = haskey(model, symbol) ? isconst(getindex(model, symbol)) : false

firstindex(model::FactorGraphModel, symbol::Symbol) = firstindex(model, getindex(model, symbol))
lastindex(model::FactorGraphModel, symbol::Symbol)  = lastindex(model, getindex(model, symbol))

firstindex(::FactorGraphModel, ::AbstractVariable) = typemin(Int64)
lastindex(::FactorGraphModel, ::AbstractVariable)  = typemax(Int64)

firstindex(::FactorGraphModel, variables::AbstractVector{<:AbstractVariable}) = firstindex(variables)
lastindex(::FactorGraphModel, variables::AbstractVector{<:AbstractVariable})  = lastindex(variables)

add!(model::FactorGraphModel, ::Nothing)  = nothing
add!(vardict::Dict, name::Symbol, entity) = vardict[name] = entity

function add!(model::FactorGraphModel, node::AbstractFactorNode)
    push!(model.nodes, node)
    add!(getstats(model), node)
    return node
end

function add!(model::FactorGraphModel, randomvar::RandomVariable)
    push!(model.random, randomvar)
    add!(getvardict(model), name(randomvar), randomvar)
    return randomvar
end

function add!(model::FactorGraphModel, constvar::ConstVariable)
    push!(model.constant, constvar)
    add!(getvardict(model), name(constvar), constvar)
    return constvar
end

function add!(model::FactorGraphModel, datavar::DataVariable)
    push!(model.data, datavar)
    add!(getvardict(model), name(datavar), datavar)
    return datavar
end

function add!(model::FactorGraphModel, collection::Tuple)
    foreach((d) -> add!(model, d), collection)
    return collection
end

function add!(model::FactorGraphModel, array::AbstractArray)
    foreach((d) -> add!(model, d), array)
    return array
end

function add!(model::FactorGraphModel, array::AbstractArray{<:RandomVariable})
    append!(model.random, array)
    add!(getvardict(model), name(first(array)), array)
    return array
end

function add!(model::FactorGraphModel, array::AbstractArray{<:ConstVariable})
    append!(model.constant, array)
    add!(getvardict(model), name(first(array)), array)
    return array
end

function add!(model::FactorGraphModel, array::AbstractArray{<:DataVariable})
    append!(model.data, array)
    add!(getvardict(model), name(first(array)), array)
    return array
end

function activate!(model::FactorGraphModel)
    filter!(getrandom(model)) do randomvar
        @assert degree(randomvar) !== 0 "Unused random variable has been found $(indexed_name(randomvar))."
        @assert degree(randomvar) !== 1 "Half-edge has been found: $(indexed_name(randomvar)). To terminate half-edges 'Uninformative' node can be used."
        return degree(randomvar) >= 2
    end

    foreach(getdata(model)) do datavar
        if !isconnected(datavar)
            @warn "Unused data variable has been found: '$(indexed_name(datavar))'. Ignore if '$(indexed_name(datavar))' has been used in deterministic nonlinear tranformation."
        end
    end

    activate!(getconstraints(model), model)
    activate!(getmeta(model), model)

    filter!(c -> isconnected(c), getconstant(model))
    foreach(r -> activate!(model, r), getrandom(model))
    foreach(n -> activate!(model, n), getnodes(model))
end

# Utility functions

## node

function node_resolve_options(model::FactorGraphModel, options::FactorNodeCreationOptions, fform, variables)
    return FactorNodeCreationOptions(
        node_resolve_factorisation(model, options, fform, variables),
        node_resolve_meta(model, options, fform, variables),
        getpipeline(options)
    )
end

## constraints 

import ReactiveMP: resolve_factorisation

node_resolve_factorisation(model::FactorGraphModel, options::FactorNodeCreationOptions, fform, variables)            = node_resolve_factorisation(model, options, factorisation(options), fform, variables)
node_resolve_factorisation(model::FactorGraphModel, options::FactorNodeCreationOptions, something, fform, variables) = something
node_resolve_factorisation(model::FactorGraphModel, options::FactorNodeCreationOptions, ::Nothing, fform, variables) = node_resolve_factorisation(model, getconstraints(model), default_factorisation(getoptions(model)), fform, variables)

node_resolve_factorisation(model::FactorGraphModel, constraints, default, fform, variables)                         = error("Cannot resolve factorisation constrains. Both `constraints` and `default_factorisation` option have been set, which is disallowed.")
node_resolve_factorisation(model::FactorGraphModel, ::ConstraintsSpecification{Tuple{}}, default, fform, variables) = default
node_resolve_factorisation(model::FactorGraphModel, ::UnspecifiedConstraints, default, fform, variables)            = default
node_resolve_factorisation(model::FactorGraphModel, constraints, ::UnspecifiedConstraints, fform, variables)        = resolve_factorisation(constraints, model, fform, variables)

node_resolve_factorisation(model::FactorGraphModel, ::ConstraintsSpecification{Tuple{}}, ::UnspecifiedConstraints, fform, variables) = resolve_factorisation(UnspecifiedConstraints(), model, fform, variables)
node_resolve_factorisation(model::FactorGraphModel, ::UnspecifiedConstraints, ::UnspecifiedConstraints, fform, variables)            = resolve_factorisation(UnspecifiedConstraints(), model, fform, variables)

## meta 

import ReactiveMP: resolve_meta

node_resolve_meta(model::FactorGraphModel, options::FactorNodeCreationOptions, fform, variables)            = node_resolve_meta(model, options, metadata(options), fform, variables)
node_resolve_meta(model::FactorGraphModel, options::FactorNodeCreationOptions, something, fform, variables) = something
node_resolve_meta(model::FactorGraphModel, options::FactorNodeCreationOptions, ::Nothing, fform, variables) = resolve_meta(getmeta(model), model, fform, variables)

## randomvar

function randomvar_resolve_options(model::FactorGraphModel, options::RandomVariableCreationOptions, name)
    qform, qprod = randomvar_resolve_marginal_form_prod(model, options, name)
    mform, mprod = randomvar_resolve_messages_form_prod(model, options, name)

    rprod = resolve_prod_constraint(options.prod_constraint, resolve_prod_constraint(qprod, mprod))

    qoptions = randomvar_options_set_marginal_form_constraint(options, qform)
    moptions = randomvar_options_set_messages_form_constraint(qoptions, mform)
    roptions = randomvar_options_set_prod_constraint(moptions, rprod)

    return roptions
end

## constraints

import ReactiveMP: resolve_marginal_form_prod, resolve_messages_form_prod

randomvar_resolve_marginal_form_prod(model::FactorGraphModel, options::RandomVariableCreationOptions, name)            = randomvar_resolve_marginal_form_prod(model, options, marginal_form_constraint(options), name)
randomvar_resolve_marginal_form_prod(model::FactorGraphModel, options::RandomVariableCreationOptions, something, name) = (something, nothing)
randomvar_resolve_marginal_form_prod(model::FactorGraphModel, options::RandomVariableCreationOptions, ::Nothing, name) = randomvar_resolve_marginal_form_prod(model, getconstraints(model), name)

randomvar_resolve_marginal_form_prod(model::FactorGraphModel, ::UnspecifiedConstraints, name) = (nothing, nothing)
randomvar_resolve_marginal_form_prod(model::FactorGraphModel, constraints, name)              = resolve_marginal_form_prod(constraints, model, name)

randomvar_resolve_messages_form_prod(model::FactorGraphModel, options::RandomVariableCreationOptions, name)            = randomvar_resolve_messages_form_prod(model, options, messages_form_constraint(options), name)
randomvar_resolve_messages_form_prod(model::FactorGraphModel, options::RandomVariableCreationOptions, something, name) = (something, nothing)
randomvar_resolve_messages_form_prod(model::FactorGraphModel, options::RandomVariableCreationOptions, ::Nothing, name) = randomvar_resolve_messages_form_prod(model, getconstraints(model), name)

randomvar_resolve_messages_form_prod(model::FactorGraphModel, ::UnspecifiedConstraints, name) = (nothing, nothing)
randomvar_resolve_messages_form_prod(model::FactorGraphModel, constraints, name)              = resolve_messages_form_prod(constraints, model, name)

# We extend `ReactiveMP` functionality here
import ReactiveMP: RandomVariable, DataVariable, ConstVariable
import ReactiveMP: RandomVariableCreationOptions, DataVariableCreationOptions
import ReactiveMP: randomvar, datavar, constvar, make_node

randomvar(model::FactorGraphModel, name::Symbol, args...) = randomvar(model, RandomVariableCreationOptions(), name, args...)
datavar(model::FactorGraphModel, name::Symbol, args...)   = datavar(model, DataVariableCreationOptions(Any), name, args...)

function __check_variable_existence(model::FactorGraphModel, name::Symbol)
    if haskey(getvardict(model), name)
        # Anonymous variables are allowed to be overwritten with the same name
        isanonymous(model[name]) || error("Variable named `$(name)` has been redefined")
    end
end

function randomvar(model::FactorGraphModel, options::RandomVariableCreationOptions, name::Symbol, args...)
    __check_variable_existence(model, name)
    return add!(model, randomvar(randomvar_resolve_options(model, options, name), name, args...))
end

function datavar(model::FactorGraphModel, options::DataVariableCreationOptions, name::Symbol, args...)
    __check_variable_existence(model, name)
    return add!(model, datavar(options, name, args...))
end

function constvar(model::FactorGraphModel, name::Symbol, args...)
    __check_variable_existence(model, name)
    return add!(model, constvar(name, args...))
end

as_variable(model::FactorGraphModel, x)        = add!(model, as_variable(x))
as_variable(model::FactorGraphModel, t::Tuple) = map((d) -> as_variable(model, d), t)

as_variable(model::FactorGraphModel, v::AbstractVariable) = v
as_variable(model::FactorGraphModel, v::AbstractVector{<:AbstractVariable}) = v

## node creation

function make_node(model::FactorGraphModel, options::FactorNodeCreationOptions, fform, args...)
    return add!(model, make_node(fform, node_resolve_options(model, options, fform, args), args...))
end

## AutoVar 

struct AutoVar
    name::Symbol
end

name(autovar::AutoVar) = autovar.name

# This function either returns a saved version of a variable from `vardict`
# Or creates a new one in two cases:
# - variable did not exist before
# - variable exists, but has been declared as anyonymous (only if `rewrite_anonymous` argument is set to true)
function make_autovar(
    model::FactorGraphModel,
    options::RandomVariableCreationOptions,
    name::Symbol,
    rewrite_anonymous::Bool = true
)
    if haskey(getvardict(model), name)
        var = model[name]
        if rewrite_anonymous && isanonymous(var)
            return ReactiveMP.randomvar(model, options, name)
        else
            return var
        end
    else
        return ReactiveMP.randomvar(model, options, name)
    end
end

function ReactiveMP.make_node(
    model::FactorGraphModel,
    options::FactorNodeCreationOptions,
    fform,
    autovar::AutoVar,
    args::Vararg{<:ReactiveMP.AbstractVariable}
)
    proxy     = isdeterministic(sdtype(fform)) ? args : nothing
    rvoptions = ReactiveMP.randomvar_options_set_proxy_variables(ReactiveMP.EmptyRandomVariableCreationOptions, proxy)
    var       = ReactiveMP.make_autovar(model, rvoptions, ReactiveMP.name(autovar), true) # add! is inside
    node      = ReactiveMP.make_node(model, options, fform, var, args...) # add! is inside
    return node, var
end

__fform_const_apply(::Type{T}, args...) where {T} = T(args...)
__fform_const_apply(f::F, args...) where {F <: Function} = f(args...)

function ReactiveMP.make_node(
    model::FactorGraphModel,
    options::FactorNodeCreationOptions,
    fform,
    autovar::AutoVar,
    args::Vararg{<:ReactiveMP.ConstVariable}
)
    if isstochastic(sdtype(fform))
        var  = ReactiveMP.make_autovar(model, ReactiveMP.EmptyRandomVariableCreationOptions, ReactiveMP.name(autovar), true)
        node = ReactiveMP.make_node(model, options, fform, var, args...) # add! is inside
        return node, var
    else
        var = add!(
            model,
            ReactiveMP.constvar(
                ReactiveMP.name(autovar),
                __fform_const_apply(fform, map((d) -> ReactiveMP.getconst(d), args)...)
            )
        )
        return nothing, var
    end
end