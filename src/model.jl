
export FactorGraphModel, create_model, model_name
export AutoVar
export getoptions, getconstraints, getmeta
export getnodes, getvariables, getrandom, getconstant, getdata

import Base: push!, show, getindex, haskey, firstindex, lastindex
import ReactiveMP: AbstractFactorNode

function create_model end

function model_name end

# Model Inference Options

"""
    ModelInferenceOptions(; kwargs...)

Creates model inference options object. The list of available options is present below.

### Options

- `limit_stack_depth`: limits the stack depth for computing messages, helps with `StackOverflowError` for some huge models, but reduces the performance of inference backend. Accepts integer as an argument that specifies the maximum number of recursive depth. Lower is better for stack overflow error, but worse for performance.

### Advanced options

- `pipeline`: changes the default pipeline for each factor node in the graph
- `global_reactive_scheduler`: changes the scheduler of reactive streams, see Rocket.jl for more info, defaults to no scheduler

See also: [`inference`](@ref), [`rxinference`](@ref)
"""
struct ModelInferenceOptions{P, S}
    pipeline                  :: P
    global_reactive_scheduler :: S
end

UnspecifiedModelInferenceOptions() = convert(ModelInferenceOptions, (;))

import Base: convert

function Base.convert(::Type{ModelInferenceOptions}, options::Nothing)
    return UnspecifiedModelInferenceOptions()
end

function Base.convert(::Type{ModelInferenceOptions}, options::NamedTuple{keys}) where {keys}
    available_options = (:pipeline, :global_reactive_scheduler, :limit_stack_depth)

    for key in keys
        key ∈ available_options || error("Unknown model inference options: $(key).")
    end

    pipeline                  = nothing
    global_reactive_scheduler = nothing

    if haskey(options, :pipeline)
        pipeline = options[:pipeline]
    end

    if haskey(options, :global_reactive_scheduler) && haskey(options, :limit_stack_depth)
        @warn "Model options have `global_reactive_scheduler` and `limit_stack_depth` options specified together. Ignoring `limit_stack_depth`."
    end

    if haskey(options, :global_reactive_scheduler)
        global_reactive_scheduler = options[:global_reactive_scheduler]
    elseif haskey(options, :limit_stack_depth)
        global_reactive_scheduler = LimitStackScheduler(options[:limit_stack_depth]...)
    end

    return ModelInferenceOptions(pipeline, global_reactive_scheduler)
end

const DefaultModelInferenceOptions = UnspecifiedModelInferenceOptions()

global_reactive_scheduler(options::ModelInferenceOptions) = something(options.global_reactive_scheduler, AsapScheduler())
get_pipeline_stages(options::ModelInferenceOptions)       = something(options.pipeline, EmptyPipelineStage())

struct FactorGraphModel{Constrains, Meta, Options <: ModelInferenceOptions}
    constraints :: Constrains
    meta        :: Meta
    options     :: Options
    nodes       :: FactorNodesCollection
    variables   :: VariablesCollection
end

Base.show(io::IO, ::Type{<:FactorGraphModel}) = print(io, "FactorGraphModel")
Base.show(io::IO, model::FactorGraphModel)    = print(io, "FactorGraphModel()")

FactorGraphModel() = FactorGraphModel(DefaultConstraints, DefaultMeta, DefaultModelInferenceOptions)

function FactorGraphModel(constraints::C, meta::M, options::O) where {C, M, O}
    return FactorGraphModel{C, M, O}(constraints, meta, options, FactorNodesCollection(), VariablesCollection())
end

getconstraints(model::FactorGraphModel) = model.constraints
getmeta(model::FactorGraphModel)        = model.meta
getoptions(model::FactorGraphModel)     = model.options
getnodes(model::FactorGraphModel)       = model.nodes
getvariables(model::FactorGraphModel)   = model.variables

import ReactiveMP: getrandom, getconstant, getdata, getvardict, getprocess

ReactiveMP.getrandom(model::FactorGraphModel)   = getrandom(getvariables(model))
ReactiveMP.getconstant(model::FactorGraphModel) = getconstant(getvariables(model))
ReactiveMP.getdata(model::FactorGraphModel)     = getdata(getvariables(model))
ReactiveMP.getvardict(model::FactorGraphModel)  = getvardict(getvariables(model))
#add process
ReactiveMP.getprocess(model::FactorGraphModel)  = getprocess(getvariables(model)) 

function Base.getindex(model::FactorGraphModel, symbol::Symbol)
    return getindex(getvariables(model), symbol)
end

function Base.haskey(model::FactorGraphModel, symbol::Symbol)
    return haskey(getvariables(model), symbol)
end

Base.broadcastable(model::FactorGraphModel) = (model,)

import ReactiveMP: hasrandomvar, hasdatavar, hasconstvar, hasprocess

ReactiveMP.hasrandomvar(model::FactorGraphModel, symbol::Symbol) = hasrandomvar(getvariables(model), symbol)
ReactiveMP.hasdatavar(model::FactorGraphModel, symbol::Symbol)   = hasdatavar(getvariables(model), symbol)
ReactiveMP.hasconstvar(model::FactorGraphModel, symbol::Symbol)  = hasconstvar(getvariables(model), symbol)
#add random process 
ReactiveMP.hasprocess(model::FactorGraphModel, symbol::Symbol)   = hasprocess(getvariables(model), symbol) #checked 

Base.firstindex(model::FactorGraphModel, symbol::Symbol) = firstindex(getvariables(model), symbol)
Base.lastindex(model::FactorGraphModel, symbol::Symbol)  = lastindex(getvariables(model), symbol)

Base.push!(::FactorGraphModel, ::Nothing) = nothing

Base.push!(model::FactorGraphModel, node::AbstractFactorNode)   = push!(getnodes(model), node)
Base.push!(model::FactorGraphModel, variable::AbstractVariable) = push!(getvariables(model), variable)

Base.push!(model::FactorGraphModel, nodes::AbstractArray{N}) where {N <: AbstractFactorNode}   = push!(getnodes(model), nodes)
Base.push!(model::FactorGraphModel, variables::AbstractArray{V}) where {V <: AbstractVariable} = push!(getvariables(model), variables)

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
    #add process 
    filter!(getprocess(model)) do randomprocess
        @assert degree(randomprocess) !== 0 "Unused random variable has been found $(indexed_name(randomprocess))."
        @assert degree(randomprocess) !== 1 "Half-edge has been found: $(indexed_name(randomprocess)). To terminate half-edges 'Uninformative' node can be used."
        return degree(randomprocess) >= 2
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
    foreach(p -> activate!(p, gscheduler), getprocess(model)) # add process 
    foreach(n -> activate!(n, gpipelinestages, gscheduler), getnodes(model))
end

## constraints 

import ReactiveMP: resolve_factorisation

node_resolve_factorisation(model::FactorGraphModel, something, fform, variables) = something
node_resolve_factorisation(model::FactorGraphModel, ::Nothing, fform, variables) = node_resolve_constraints_factorisation(model, getconstraints(model), fform, variables)

node_resolve_constraints_factorisation(model::FactorGraphModel, constraints, fform, variables)                         = resolve_factorisation(constraints, getvariables(model), fform, variables)
node_resolve_constraints_factorisation(model::FactorGraphModel, ::ConstraintsSpecification{Tuple{}}, fform, variables) = resolve_factorisation(UnspecifiedConstraints(), getvariables(model), fform, variables)

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
# random process 
import ReactiveMP: randomprocess_options_set_marginal_form_check_strategy, randomprocess_options_set_marginal_form_constraint
import ReactiveMP: randomprocess_options_set_messages_form_check_strategy, randomprocess_options_set_messages_form_constraint
import ReactiveMP: randomprocess_options_set_pipeline, randomprocess_options_set_prod_constraint
import ReactiveMP: randomprocess_options_set_prod_strategy, randomprocess_options_set_proxy_variables

function randomprocess_resolve_options(model::FactorGraphModel, options::RandomProcessCreationOptions, name)
    qform, qprod = randomprocess_resolve_marginal_form_prod(model, options, name)
    mform, mprod = randomprocess_resolve_messages_form_prod(model, options, name)

    rprod = resolve_prod_constraint(options.prod_constraint, resolve_prod_constraint(qprod, mprod)) #check this 

    qoptions = randomprocess_options_set_marginal_form_constraint(options, qform)
    moptions = randomprocess_options_set_messages_form_constraint(qoptions, mform)
    roptions = randomprocess_options_set_prod_constraint(moptions, rprod)

    return roptions
end
# Model Generator

"""
    ModelGenerator

`ModelGenerator` is a special object that is used in the `inference` function to lazily create model later on given `constraints`, `meta` and `options`.

See also: [`inference`](@ref)
"""
struct ModelGenerator{G, A, K}
    generator :: G
    args      :: A
    kwargs    :: K

    ModelGenerator(generator::G, args::A, kwargs::K) where {G, A, K} = new{G, A, K}(generator, args, kwargs)
end

function (generator::ModelGenerator)(; constraints = nothing, meta = nothing, options = nothing)
    sconstraints = something(constraints, UnspecifiedConstraints())
    smeta        = something(meta, UnspecifiedMeta())
    soptions     = something(options, UnspecifiedModelInferenceOptions())
    return generator(FactorGraphModel(sconstraints, smeta, soptions))
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
function create_model(generator::ModelGenerator; constraints = nothing, meta = nothing, options = nothing)
    sconstraints = something(constraints, UnspecifiedConstraints())
    smeta        = something(meta, UnspecifiedMeta())
    soptions     = something(options, UnspecifiedModelInferenceOptions())
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

## constraints randomprocess 

randomprocess_resolve_marginal_form_prod(model::FactorGraphModel, options::RandomProcessCreationOptions, name)            = randomprocess_resolve_marginal_form_prod(model, options, marginal_form_constraint(options), name)
randomprocess_resolve_marginal_form_prod(model::FactorGraphModel, options::RandomProcessCreationOptions, something, name) = (something, nothing)
randomprocess_resolve_marginal_form_prod(model::FactorGraphModel, options::RandomProcessCreationOptions, ::Nothing, name) = randomprocess_resolve_marginal_form_prod(model, getconstraints(model), name)

randomprocess_resolve_marginal_form_prod(model::FactorGraphModel, ::UnspecifiedConstraints, name) = (nothing, nothing)
randomprocess_resolve_marginal_form_prod(model::FactorGraphModel, constraints, name)              = resolve_marginal_form_prod(constraints, name)  

randomprocess_resolve_messages_form_prod(model::FactorGraphModel, options::RandomProcessCreationOptions, name)            = randomprocess_resolve_messages_form_prod(model, options, messages_form_constraint(options), name)
randomprocess_resolve_messages_form_prod(model::FactorGraphModel, options::RandomProcessCreationOptions, something, name) = (something, nothing)
randomprocess_resolve_messages_form_prod(model::FactorGraphModel, options::RandomProcessCreationOptions, ::Nothing, name) = randomprocess_resolve_messages_form_prod(model, getconstraints(model), name)

randomprocess_resolve_messages_form_prod(model::FactorGraphModel, ::UnspecifiedConstraints, name) = (nothing, nothing)
randomprocess_resolve_messages_form_prod(model::FactorGraphModel, constraints, name)              = resolve_messages_form_prod(constraints, name)

# We extend `ReactiveMP` functionality here
import ReactiveMP: RandomVariable, DataVariable, ConstVariable, RandomProcess
import ReactiveMP: RandomVariableCreationOptions, DataVariableCreationOptions, RandomProcessCreationOptions
import ReactiveMP: randomvar, datavar, constvar, make_node, randomprocess

ReactiveMP.randomvar(model::FactorGraphModel, name::Symbol, args...) = randomvar(model, RandomVariableCreationOptions(), name, args...)
ReactiveMP.datavar(model::FactorGraphModel, name::Symbol, args...)   = datavar(model, DataVariableCreationOptions(Any), name, args...)
## add random process 
ReactiveMP.randomprocess(model::FactorGraphModel, name::Symbol, args...) = randomprocess(model, RandomProcessCreationOptions(), name, args...)

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

#add randomprocess 
function ReactiveMP.randomprocess(model::FactorGraphModel, options::RandomProcessCreationOptions, name::Symbol, args...)
    __check_variable_existence(model, name)
    return push!(model, randomprocess(randomprocess_resolve_options(model, options, name), name, args...))
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
        node_resolve_factorisation(model, factorisation(options), fform, variables), node_resolve_meta(model, metadata(options), fform, variables), getpipeline(options)
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

function ReactiveMP.make_node(model::FactorGraphModel, options::FactorNodeCreationOptions, fform, autovar::AutoVar, args::Vararg)
    foreach(args) do arg
        @assert (typeof(arg) <: AbstractVariable || eltype(arg) <: AbstractVariable) "`make_node` cannot create a node with the given arguments autovar = $(autovar), args = [ $(args...) ]"
    end

    proxy     = isdeterministic(sdtype(fform)) ? args : nothing
    rvoptions = ReactiveMP.randomvar_options_set_proxy_variables(ReactiveMP.EmptyRandomVariableCreationOptions, proxy)
    var       = make_autovar(model, rvoptions, ReactiveMP.name(autovar), true) # add! is inside
    node      = ReactiveMP.make_node(model, options, fform, var, args...) # add! is inside
    return node, var
end

__fform_const_apply(::Type{T}, args...) where {T} = T(args...)
__fform_const_apply(f::F, args...) where {F <: Function} = f(args...)

function ReactiveMP.make_node(model::FactorGraphModel, options::FactorNodeCreationOptions, fform, autovar::AutoVar, args::Vararg{<:ReactiveMP.ConstVariable})
    if isstochastic(sdtype(fform))
        var  = make_autovar(model, ReactiveMP.EmptyRandomVariableCreationOptions, ReactiveMP.name(autovar), true)
        node = ReactiveMP.make_node(model, options, fform, var, args...) # add! is inside
        return node, var
    else
        var = push!(model, ReactiveMP.constvar(ReactiveMP.name(autovar), __fform_const_apply(fform, map((d) -> ReactiveMP.getconst(d), args)...)))
        return nothing, var
    end
end

## AutoNode 

struct AutoNode end

Base.broadcastable(::AutoNode) = Ref(AutoNode())

function ReactiveMP.make_node(model::FactorGraphModel, ::AutoNode, autovar::AutoVar, distribution::Distribution)
    var = make_autovar(model, ReactiveMP.EmptyRandomVariableCreationOptions, ReactiveMP.name(autovar), true)
    return ReactiveMP.make_node(model, AutoNode(), var, distribution)
end

function ReactiveMP.make_node(model::FactorGraphModel, ::AutoNode, var::RandomVariable, distribution::Distribution)
    args = map(v -> as_variable(model, v), params(distribution))
    node = ReactiveMP.make_node(model, FactorNodeCreationOptions(), typeof(distribution), var, args...)
    return node, var
end
