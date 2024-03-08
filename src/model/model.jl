
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