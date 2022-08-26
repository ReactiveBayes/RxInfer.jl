export AbstractModelSpecification
export ModelOptions, model_options
export FactorGraphModel, create_model, model_name
export AutoVar
export getoptions, getconstraints, getmeta, getstats
export getnodes, getrandom, getconstant, getdata

import Base: push!, show, getindex, haskey, firstindex, lastindex
import ReactiveMP: AbstractFactorNode

function create_model end

function model_name end

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

UnspecifiedModelOptions() = model_options()

struct FactorGraphModel{C, M, O}
    constraints :: C
    meta        :: M
    options     :: O
    nodes       :: FactorNodesCollection
    variables   :: VariablesCollection
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

function FactorGraphModel(constraints::C, meta::M, options::O) where { C, M, O }
    return FactorGraphModel{C, M, O}(constraints, meta, options, FactorNodesCollection(), VariablesCollection())
end

getconstraints(model::FactorGraphModel) = model.constraints
getmeta(model::FactorGraphModel)        = model.meta
getoptions(model::FactorGraphModel)     = model.options
getnodes(model::FactorGraphModel)       = model.nodes
getvariables(model::FactorGraphModel)   = model.variables

import ReactiveMP: getrandom, getconstant, getdata, getvardict

ReactiveMP.getrandom(model::FactorGraphModel)   = getrandom(getvariables(model))
ReactiveMP.getconstant(model::FactorGraphModel) = getconstant(getvariables(model))
ReactiveMP.getdata(model::FactorGraphModel)     = getdata(getvariables(model))
ReactiveMP.getvardict(model::FactorGraphModel)  = getvardict(getvariables(model))

function Base.getindex(model::FactorGraphModel, symbol::Symbol)
    return getindex(getvariables(model), symbol)
end

function Base.haskey(model::FactorGraphModel, symbol::Symbol)
    return haskey(getvariables(model), symbol)
end

Base.broadcastable(model::FactorGraphModel) = (model, )

import ReactiveMP: hasrandomvar, hasdatavar, hasconstvar

ReactiveMP.hasrandomvar(model::FactorGraphModel, symbol::Symbol) = hasrandomvar(getvariables(model), symbol)
ReactiveMP.hasdatavar(model::FactorGraphModel, symbol::Symbol)   = hasdatavar(getvariables(model), symbol)
ReactiveMP.hasconstvar(model::FactorGraphModel, symbol::Symbol)  = hasconstvar(getvariables(model), symbol)

Base.firstindex(model::FactorGraphModel, symbol::Symbol) = firstindex(getvariables(model), symbol)
Base.lastindex(model::FactorGraphModel, symbol::Symbol)  = lastindex(getvariables(model), symbol)

Base.push!(::FactorGraphModel, ::Nothing) = nothing

Base.push!(model::FactorGraphModel, node::AbstractFactorNode)   = push!(getnodes(model), node)
Base.push!(model::FactorGraphModel, variable::AbstractVariable) = push!(getvariables(model), variable)

Base.push!(model::FactorGraphModel, nodes::AbstractArray{N}) where { N <: AbstractFactorNode }   = push!(getnodes(model), nodes)
Base.push!(model::FactorGraphModel, variables::AbstractArray{V}) where { V <: AbstractVariable } = push!(getvariables(model), variables)

function Base.push!(model::FactorGraphModel, collection::Tuple)
    foreach((d) -> push!(model, d), collection)
    return collection
end

function Base.push!(model::FactorGraphModel, array::AbstractArray)
    foreach((d) -> add!(model, d), array)
    return array
end

import ReactiveMP: activate!

function ReactiveMP.activate!(model::FactorGraphModel)
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

    activate!(getconstraints(model), getnodes(model), getvariables(model))
    activate!(getmeta(model), getnodes(model), getvariables(model))

    gpipelinestages = get_pipeline_stages(getoptions(model))
    gscheduler      = global_reactive_scheduler(getoptions(model))

    filter!(c -> isconnected(c), getconstant(model))
    foreach(r -> activate!(r, gscheduler), getrandom(model))
    foreach(n -> activate!(n, gpipelinestages, gscheduler), getnodes(model))
end

## constraints 

import ReactiveMP: resolve_factorisation

node_resolve_factorisation(model::FactorGraphModel, something, fform, variables) = something
node_resolve_factorisation(model::FactorGraphModel, ::Nothing, fform, variables) = node_resolve_factorisation(model, getconstraints(model), default_factorisation(getoptions(model)), fform, variables)

node_resolve_factorisation(model::FactorGraphModel, constraints, default, fform, variables)                         = error("Cannot resolve factorisation constrains. Both `constraints` and `default_factorisation` option have been set, which is disallowed.")
node_resolve_factorisation(model::FactorGraphModel, ::ConstraintsSpecification{Tuple{}}, default, fform, variables) = default
node_resolve_factorisation(model::FactorGraphModel, ::UnspecifiedConstraints, default, fform, variables)            = default
node_resolve_factorisation(model::FactorGraphModel, constraints, ::UnspecifiedConstraints, fform, variables)        = resolve_factorisation(constraints, getvariables(model), fform, variables)

node_resolve_factorisation(model::FactorGraphModel, ::ConstraintsSpecification{Tuple{}}, ::UnspecifiedConstraints, fform, variables) = resolve_factorisation(UnspecifiedConstraints(), getvariables(model), fform, variables)
node_resolve_factorisation(model::FactorGraphModel, ::UnspecifiedConstraints, ::UnspecifiedConstraints, fform, variables)            = resolve_factorisation(UnspecifiedConstraints(), getvariables(model), fform, variables)

## meta 

import ReactiveMP: resolve_meta

node_resolve_meta(model::FactorGraphModel, something, fform, variables) = something
node_resolve_meta(model::FactorGraphModel, ::Nothing, fform, variables) = resolve_meta(getmeta(model), fform, variables)

## randomvar

import ReactiveMP: randomvar_options_set_marginal_form_check_strategy, randomvar_options_set_marginal_form_constraint
import ReactiveMP: randomvar_options_set_messages_form_check_strategy, randomvar_options_set_messages_form_constraint
import ReactiveMP: randomvar_options_set_pipeline, randomvar_options_set_prod_constraint
import ReactiveMP: randomvar_options_set_prod_strategy, randomvar_options_set_proxy_variables

function randomvar_resolve_options(model::FactorGraphModel, options::RandomVariableCreationOptions, name)
    qform, qprod = randomvar_resolve_marginal_form_prod(model, options, name)
    mform, mprod = randomvar_resolve_messages_form_prod(model, options, name)

    rprod = resolve_prod_constraint(options.prod_constraint, resolve_prod_constraint(qprod, mprod))

    qoptions = randomvar_options_set_marginal_form_constraint(options, qform)
    moptions = randomvar_options_set_messages_form_constraint(qoptions, mform)
    roptions = randomvar_options_set_prod_constraint(moptions, rprod)

    return roptions
end

# Model Generator

"""
    ModelGenerator

`ModelGenerator` is a special object that is used in the `inference` function to lazily create model later on given `constraints`, `meta` and `options`.

See also: [`inference`](@ref)
"""
struct ModelGenerator{G, A, K}
    generator   :: G
    args        :: A
    kwargs      :: K

    ModelGenerator(generator::G, args::A, kwargs::K) where {G, A, K} = new{G, A, K}(generator, args, kwargs)
end

function (generator::ModelGenerator)(; constraints = nothing, meta = nothing, options = nothing)
    return generator(FactorGraphModel(constraints, meta, options))
end

function (generator::ModelGenerator)(model::FactorGraphModel)
    return generator.generator(model, generator.args...; generator.kwargs...)
end

"""
    create_model(::ModelGenerator, constraints = nothing, meta = nothing, options = nothing)

Creates an instance of `FactorGraphModel` from the given model specification as well as optional `constraints`, `meta` and `options`.

Returns a tuple of 2 values:
- 1. an instance of `FactorGraphModel`
- 2. return value from the `@model` macro function definition
"""
function create_model(generator::ModelGenerator, constraints = nothing, meta = nothing, options = nothing)
    sconstraints = something(constraints, UnspecifiedConstraints())
    smeta        = something(meta, UnspecifiedMeta())
    soptions     = something(options, UnspecifiedModelOptions())
    model        = FactorGraphModel(sconstraints, smeta, soptions)
    returnvars   = generator(model)
    # `activate!` function creates reactive connections in the factor graph model and finalises model structure
    activate!(model)
    return model, returnvars
end

## constraints

import ReactiveMP: marginal_form_constraint, messages_form_constraint, prod_constraint
import ReactiveMP: resolve_marginal_form_prod, resolve_messages_form_prod

randomvar_resolve_marginal_form_prod(model::FactorGraphModel, options::RandomVariableCreationOptions, name)            = randomvar_resolve_marginal_form_prod(model, options, marginal_form_constraint(options), name)
randomvar_resolve_marginal_form_prod(model::FactorGraphModel, options::RandomVariableCreationOptions, something, name) = (something, nothing)
randomvar_resolve_marginal_form_prod(model::FactorGraphModel, options::RandomVariableCreationOptions, ::Nothing, name) = randomvar_resolve_marginal_form_prod(model, getconstraints(model), name)

randomvar_resolve_marginal_form_prod(model::FactorGraphModel, ::UnspecifiedConstraints, name) = (nothing, nothing)
randomvar_resolve_marginal_form_prod(model::FactorGraphModel, constraints, name)              = resolve_marginal_form_prod(constraints, name)

randomvar_resolve_messages_form_prod(model::FactorGraphModel, options::RandomVariableCreationOptions, name)            = randomvar_resolve_messages_form_prod(model, options, messages_form_constraint(options), name)
randomvar_resolve_messages_form_prod(model::FactorGraphModel, options::RandomVariableCreationOptions, something, name) = (something, nothing)
randomvar_resolve_messages_form_prod(model::FactorGraphModel, options::RandomVariableCreationOptions, ::Nothing, name) = randomvar_resolve_messages_form_prod(model, getconstraints(model), name)

randomvar_resolve_messages_form_prod(model::FactorGraphModel, ::UnspecifiedConstraints, name) = (nothing, nothing)
randomvar_resolve_messages_form_prod(model::FactorGraphModel, constraints, name)              = resolve_messages_form_prod(constraints, name)

# We extend `ReactiveMP` functionality here
import ReactiveMP: RandomVariable, DataVariable, ConstVariable
import ReactiveMP: RandomVariableCreationOptions, DataVariableCreationOptions
import ReactiveMP: randomvar, datavar, constvar, make_node

ReactiveMP.randomvar(model::FactorGraphModel, name::Symbol, args...) = randomvar(model, RandomVariableCreationOptions(), name, args...)
ReactiveMP.datavar(model::FactorGraphModel, name::Symbol, args...)   = datavar(model, DataVariableCreationOptions(Any), name, args...)

function __check_variable_existence(model::FactorGraphModel, name::Symbol)
    if haskey(getvardict(model), name)
        # Anonymous variables are allowed to be overwritten with the same name
        isanonymous(model[name]) || error("Variable named `$(name)` has been redefined")
    end
end

function ReactiveMP.randomvar(model::FactorGraphModel, options::RandomVariableCreationOptions, name::Symbol, args...)
    __check_variable_existence(model, name)
    return push!(model, randomvar(randomvar_resolve_options(model, options, name), name, args...))
end

function ReactiveMP.datavar(model::FactorGraphModel, options::DataVariableCreationOptions, name::Symbol, args...)
    __check_variable_existence(model, name)
    return push!(model, datavar(options, name, args...))
end

function ReactiveMP.constvar(model::FactorGraphModel, name::Symbol, args...)
    __check_variable_existence(model, name)
    return push!(model, constvar(name, args...))
end

import ReactiveMP: as_variable, undo_as_variable

ReactiveMP.as_variable(model::FactorGraphModel, x)        = push!(model, ReactiveMP.as_variable(x))
ReactiveMP.as_variable(model::FactorGraphModel, t::Tuple) = map((d) -> ReactiveMP.as_variable(model, d), t)

ReactiveMP.as_variable(model::FactorGraphModel, v::AbstractVariable) = v
ReactiveMP.as_variable(model::FactorGraphModel, v::AbstractVector{<:AbstractVariable}) = v

## node creation

import ReactiveMP: make_node, FactorNodeCreationOptions
import ReactiveMP: factorisation, metadata, getpipeline

function node_resolve_options(model::FactorGraphModel, options, fform, variables)
    return FactorNodeCreationOptions(
        node_resolve_factorisation(model, factorisation(options), fform, variables),
        node_resolve_meta(model, metadata(options), fform, variables),
        getpipeline(options)
    )
end

function ReactiveMP.make_node(model::FactorGraphModel, options::FactorNodeCreationOptions, fform, args...)
    return push!(model, make_node(fform, node_resolve_options(model, options, fform, args), args...))
end

## AutoVar 

import ReactiveMP: name

struct AutoVar
    name::Symbol
end

ReactiveMP.name(autovar::AutoVar) = autovar.name

# This function either returns a saved version of a variable from `vardict`
# Or creates a new one in two cases:
# - variable did not exist before
# - variable exists, but has been declared as anyonymous (only if `rewrite_anonymous` argument is set to true)
function make_autovar(model::FactorGraphModel, options::RandomVariableCreationOptions, name::Symbol, rewrite_anonymous::Bool = true)
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
    var       = make_autovar(model, rvoptions, ReactiveMP.name(autovar), true) # add! is inside
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
        var  = make_autovar(model, ReactiveMP.EmptyRandomVariableCreationOptions, ReactiveMP.name(autovar), true)
        node = ReactiveMP.make_node(model, options, fform, var, args...) # add! is inside
        return node, var
    else
        var = push!(model, ReactiveMP.constvar(ReactiveMP.name(autovar), __fform_const_apply(fform, map((d) -> ReactiveMP.getconst(d), args)...)))
        return nothing, var
    end
end
