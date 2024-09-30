
export ProbabilisticModel, UnfactorizedData
export getmodel, getreturnval, getvardict, getrandomvars, getconstantvars, getdatavars, getfactornodes

import Base: push!, show, getindex, haskey, firstindex, lastindex
import ReactiveMP: getaddons, AbstractFactorNode
import GraphPPL: ModelGenerator, getmodel, getkwargs, create_model
import Rocket: getscheduler

"""
    UnfactorizedData{D}

A wrapper struct to wrap data that should not be factorized out by default during inference.
When performing Bayesian Inference with message passing, every factor node contains a local
factorization constraint on the variational posterior distribution. For data, we usually regarding
data as an independent component in the variational posterior distribution. However, in some cases,
for example when we are predicting data, we do not want to factorize out the data. In such cases,
we can wrap the data with `UnfactorizedData` struct to prevent the factorization and craft a custom
node-local factorization with the `@constraints` macro.
"""
struct UnfactorizedData{D}
    data::D
end

get_data(x) = x
get_data(x::UnfactorizedData) = x.data

"A structure that holds the factor graph representation of a probabilistic model."
struct ProbabilisticModel{M}
    model::M
end

"Returns the underlying factor graph model."
getmodel(model::ProbabilisticModel) = model.model

"Returns the value from the `return ...` operator inside the model specification."
getreturnval(model::ProbabilisticModel) = getreturnval(getmodel(model))

"Returns the (nested) dictionary of random variables from the model specification."
getvardict(model::ProbabilisticModel) = getvardict(getmodel(model))

"Returns the random variables from the model specification."
getrandomvars(model::ProbabilisticModel) = getrandomvars(getmodel(model))

"Returns the data variables from the model specification."
getdatavars(model::ProbabilisticModel) = getdatavars(getmodel(model))

"Returns the constant variables from the model specification."
getconstantvars(model::ProbabilisticModel) = getconstantvars(getmodel(model))

"Returns the factor nodes from the model specification."
getfactornodes(model::ProbabilisticModel) = getfactornodes(getmodel(model))

# Redirect the `getvarref` call to the underlying model
getvarref(model::ProbabilisticModel, label) = getvarref(getmodel(model), label)

"""
    ConditionedModelGenerator(generator, conditioned_on)

Accepts a model generator and data to condition on. 
The `generator` must be `GraphPPL.ModelGenerator` object.
The `conditioned_on` must be named tuple or a dictionary with keys corresponding to the names of the input arguments in the model.
"""
struct ConditionedModelGenerator{G, D}
    generator::G
    conditioned_on::D
end

getgenerator(generator::ConditionedModelGenerator) = generator.generator
getconditioned_on(generator::ConditionedModelGenerator) = generator.conditioned_on

"""
    condition_on(generator::ModelGenerator; kwargs...)

A function that creates a `ConditionedModelGenerator` object from `GraphPPL.ModelGenerator`.
The `|` operator can be used as a shorthand for this function.

```jldoctest
julia> using RxInfer

julia> @model function beta_bernoulli(y, a, b)
           θ ~ Beta(a, b)
           y .~ Bernoulli(θ)
       end

julia> conditioned_model = beta_bernoulli(a = 1.0, b = 2.0) | (y = [ 1.0, 0.0, 1.0 ], )
beta_bernoulli(a = 1.0, b = 2.0) conditioned on: 
  y = [1.0, 0.0, 1.0]

julia> RxInfer.create_model(conditioned_model) isa RxInfer.ProbabilisticModel
true
```
"""
function condition_on(generator::ModelGenerator; kwargs...)
    return ConditionedModelGenerator(generator, NamedTuple(kwargs))
end

function condition_on(generator::ModelGenerator, data)
    return ConditionedModelGenerator(generator, data)
end

"""
An alias for [`RxInfer.condition_on`](@ref).
"""
function Base.:(|)(generator::ModelGenerator, data)
    return condition_on(generator, data)
end

function Base.show(io::IO, generator::ConditionedModelGenerator)
    print(io, getmodel(getgenerator(generator)), "(")
    print(io, join(Iterators.map(kv -> string(kv[1], " = ", kv[2]), getkwargs(getgenerator(generator))), ", "))
    print(io, ")")
    if !isnothing(getconditioned_on(generator))
        println(io, " conditioned on: ")
        foreach(keys(getconditioned_on(generator))) do key
            println(io, "  ", key, " = ", getconditioned_on(generator)[key])
        end
    end
end

"""
    create_model(generator::ConditionedModelGenerator)

Materializes the model specification conditioned on some data into a corresponding factor graph representation.
Returns [`ProbabilisticModel`](@ref).
"""
function create_model(generator::ConditionedModelGenerator)
    return __infer_create_factor_graph_model(getgenerator(generator), getconditioned_on(generator))
end

function __infer_create_factor_graph_model(::ModelGenerator, conditioned_on)
    error("Cannot create a factor graph model from a `ModelGenerator` object. The `data` object must be a `NamedTuple` or a `Dict`. Got `$(typeof(conditioned_on))` instead.")
end

# This function works for static data, such as `NamedTuple` or a `Dict`
function __infer_create_factor_graph_model(generator::ModelGenerator, conditioned_on::Union{NamedTuple, Dict})
    # If the data is already a `NamedTuple` this should not really matter 
    # But it makes it easier to deal with the `Dict` type, which is unordered by default
    ntdata = NamedTuple(conditioned_on)::NamedTuple
    model  = create_model(generator) do model, ctx
        ikeys = keys(ntdata)
        interfaces = map(ikeys) do key
            return __infer_create_data_interface(model, ctx, key, ntdata[key])
        end
        return NamedTuple{ikeys}(interfaces)
    end
    return ProbabilisticModel(model)
end

"""
An object that is used to condition on unknown data. That may be necessary to create a model from a `ModelGenerator` object
for which data is not known at the time of the model creation. 
"""
struct DeferredDataHandler end

function Base.show(io::IO, ::DeferredDataHandler)
    print(io, "[ deffered data ]")
end

# We use the `datalabel` to instantiate the data interface for the model, in case of `DeferredDataHandler`
# the data is not known at the time of the model creation
function __infer_create_data_interface(model, context, key::Symbol, ::DeferredDataHandler)
    return GraphPPL.datalabel(model, context, GraphPPL.NodeCreationOptions(kind = :data, factorized = true), key, GraphPPL.MissingCollection())
end

# In all other cases we use the `datalabel` to instantiate the data interface for the model and the data is known at the time of the model creation
function __infer_create_data_interface(model, context, key::Symbol, data::UnfactorizedData{D}) where {D}
    return GraphPPL.datalabel(model, context, GraphPPL.NodeCreationOptions(kind = :data, factorized = false), key, get_data(data))
end

# In all other cases we use the `datalabel` to instantiate the data interface for the model and the data is known at the time of the model creation
function __infer_create_data_interface(model, context, key::Symbol, data)
    return GraphPPL.datalabel(model, context, GraphPPL.NodeCreationOptions(kind = :data, factorized = true), key, data)
end

merge_data_handlers(data::Dict, newdata::Dict) = merge(data, newdata)
merge_data_handlers(data::Dict, newdata::NamedTuple) = merge(data, convert(Dict, newdata))
merge_data_handlers(data::NamedTuple, newdata::Dict) = merge(convert(Dict, data), newdata)
merge_data_handlers(data::NamedTuple, newdata::NamedTuple) = merge(data, newdata)

# This function creates a named tuple of `DeferredDataHandler` objects from a tuple of symbols
function create_deferred_data_handlers(symbols::NTuple{N, Symbol}) where {N}
    return NamedTuple{symbols}(map(_ -> DeferredDataHandler(), symbols))
end

# This function creates a dictionary of `DeferredDataHandler` objects from an array of symbols
function create_deferred_data_handlers(symbols::AbstractVector{Symbol})
    return Dict{Symbol, DeferredDataHandler}(map(s -> s => DeferredDataHandler(), symbols))
end
