
export ProbabilisticModel
export getmodel, getreturnval, getvardict, getrandomvars, getconstantvars, getdatavars, getfactornodes

import Base: push!, show, getindex, haskey, firstindex, lastindex
import ReactiveMP: getaddons, AbstractFactorNode
import GraphPPL: ModelGenerator, getmodel, getkwargs, create_model
import Rocket: getscheduler

struct ProbabilisticModel{M}
    model::M
end

getmodel(model::ProbabilisticModel) = model.model

getreturnval(model::ProbabilisticModel) = getreturnval(getmodel(model))
getvardict(model::ProbabilisticModel) = getvardict(getmodel(model))
getrandomvars(model::ProbabilisticModel) = getrandomvars(getmodel(model))
getdatavars(model::ProbabilisticModel) = getdatavars(getmodel(model))
getconstantvars(model::ProbabilisticModel) = getconstantvars(getmodel(model))
getfactornodes(model::ProbabilisticModel) = getfactornodes(getmodel(model))

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
struct DefferedDataHandler end

function Base.show(io::IO, ::DefferedDataHandler)
    print(io, "[ deffered data ]")
end

# We use the `LazyIndex` to instantiate the data interface for the model, in case of `DefferedDataHandler`
# the data is not known at the time of the model creation
function __infer_create_data_interface(model, context, key::Symbol, ::DefferedDataHandler)
    return GraphPPL.getorcreate!(model, context, GraphPPL.NodeCreationOptions(kind = :data, factorized = true), key, GraphPPL.LazyIndex())
end

# In all other cases we use the `LazyIndex` to instantiate the data interface for the model and the data is known at the time of the model creation
function __infer_create_data_interface(model, context, key::Symbol, data)
    return GraphPPL.getorcreate!(model, context, GraphPPL.NodeCreationOptions(kind = :data, factorized = true), key, GraphPPL.LazyIndex(data))
end

# This function appends `DefferedDataHandler` objects to the existing data
function append_deffered_data_handlers(data, symbols)
    # Check if the data already has the data associated with a key provided in the `symbols`
    foreach(symbols) do symbol
        if haskey(data, symbol)
            error("Cannot add `DefferedDataHandler` for the key `$(symbol)`. Data has already been defined for the key `$(symbol)`")
        end
    end
    return __merge_data_handlers(data, create_deffered_data_handlers(symbols))
end

__merge_data_handlers(data::Dict, newdata::Dict) = merge(data, newdata)
__merge_data_handlers(data::Dict, newdata::NamedTuple) = merge(data, convert(Dict, newdata))
__merge_data_handlers(data::NamedTuple, newdata::Dict) = merge(convert(Dict, data), newdata)
__merge_data_handlers(data::NamedTuple, newdata::NamedTuple) = merge(data, newdata)

# This function creates a named tuple of `DefferedDataHandler` objects from a tuple of symbols
function create_deffered_data_handlers(symbols::NTuple{N, Symbol}) where {N}
    return NamedTuple{symbols}(map(_ -> DefferedDataHandler(), symbols))
end

# This function creates a dictionary of `DefferedDataHandler` objects from an array of symbols
function create_deffered_data_handlers(symbols::AbstractVector{Symbol})
    return Dict{Symbol, DefferedDataHandler}(map(s -> s => DefferedDataHandler(), symbols))
end