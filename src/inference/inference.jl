export KeepEach, KeepLast
export infer
export InferenceResult
export RxInferenceEngine, RxInferenceEvent

import DataStructures: CircularBuffer
import GraphPPL: ModelGenerator, create_model

import ReactiveMP: israndom, isdata, isconst
import ReactiveMP: CountingReal

import ProgressMeter

obtain_prediction(variable::Any) = getprediction(variable)
obtain_prediction(variables::AbstractArray) = getpredictions(variables)

obtain_marginal(variable::Any, strategy = SkipInitial()) = getmarginal(variable, strategy)
obtain_marginal(variables::AbstractArray, strategy = SkipInitial()) = getmarginals(variables, strategy)

assign_marginal!(variable::Any, marginal) = setmarginal!(variable, marginal)
assign_marginal!(variables::AbstractArray, marginals) = setmarginals!(variables, marginals)

assign_message!(variable::Any, message) = setmessage!(variable, message)
assign_message!(variables::AbstractArray, messages) = setmessages!(variables, messages)

"Instructs the inference engine to keep each marginal update for all intermediate iterations."
struct KeepEach end

"Instructs the inference engine to keep only the last marginal update and disregard intermediate updates."
struct KeepLast end

make_actor(::Any, ::KeepEach) = keep(Marginal)
make_actor(x::AbstractArray, ::KeepEach) = keep(typeof(similar(x, Marginal)))

make_actor(::Any, ::KeepEach, capacity::Integer) = circularkeep(Marginal, capacity)
make_actor(x::AbstractArray, ::KeepEach, capacity::Integer) = circularkeep(typeof(similar(x, Marginal)), capacity)

make_actor(::Any, ::KeepLast) = storage(Marginal)
make_actor(x::AbstractArray, ::KeepLast) = buffer(Marginal, size(x))

make_actor(::Any, ::KeepLast, capacity::Integer) = storage(Marginal)
make_actor(x::AbstractArray, ::KeepLast, capacity::Integer) = buffer(Marginal, size(x))

## Inference ensure update

import Rocket: Actor, on_next!, on_error!, on_complete!

# We can use `MarginalHasBeenUpdated` both as an actor in within the `ensure_update` operator
mutable struct MarginalHasBeenUpdated <: Actor{Any}
    updated::Bool
end

reset_updated!(updated::MarginalHasBeenUpdated) = updated.updated = false
set_updated!(updated::MarginalHasBeenUpdated)   = updated.updated = true

Rocket.on_next!(updated::MarginalHasBeenUpdated, anything) = set_updated!(updated)
Rocket.on_error!(updated::MarginalHasBeenUpdated, err)     = begin end
Rocket.on_complete!(updated::MarginalHasBeenUpdated)       = begin end

# This creates a `tap` operator that will set the `updated` flag to true. 
# Later on we check flags and `unset!` them after the `update!` procedure
ensure_update(model::ProbabilisticModel, callback, variable_name::Symbol, updated::MarginalHasBeenUpdated) =
    tap() do update
        set_updated!(updated)
        callback(model, variable_name, update)
    end

ensure_update(model::ProbabilisticModel, ::Nothing, variable_name::Symbol, updated::MarginalHasBeenUpdated) =
    tap() do _
        set_updated!(updated) # If `callback` is nothing we simply set updated flag
    end

function check_and_reset_updated!(updates)
    if all((v) -> v.updated, values(updates))
        foreach(reset_updated!, values(updates))
    else
        not_updated = filter((pair) -> !last(pair).updated, updates)
        names = join(keys(not_updated), ", ")
        error("""
              Variables [ $(names) ] have not been updated after an update event. 
              Therefore, make sure to initialize all required marginals and messages. See `initialization` keyword argument for the inference function. 
              See the official documentation for detailed information regarding the initialization.
              """)
    end
end

## Extra error handling

function inference_process_error(error)
    # By default, rethrow the error
    return inference_process_error(error, true)
end

function inference_process_error(error, rethrow)
    if rethrow
        Base.rethrow(error)
    end
    return error, catch_backtrace()
end

# We want to show an extra hint in case the error is of type `StackOverflowError`
function inference_process_error(err::StackOverflowError, rethrow)
    @error """
    Stack overflow error occurred during the inference procedure. 
    The inference engine may execute message update rules recursively, hence, the model graph size might be causing this error. 
    To resolve this issue, try using `limit_stack_depth` inference option for model creation. See `?inference` documentation for more details.
    The `limit_stack_depth` option does not help against over stack overflow errors that might happening outside of the model creation or message update rules execution.
    """
    if rethrow
        Base.rethrow(err) # Shows the original stack trace
    end
    return err, catch_backtrace()
end

function inference_check_itertype(::Symbol, ::Union{Nothing, Tuple, Vector})
    # This function check is the second argument is of type `Nothing`, `Tuple` or `Vector`. 
    # Does nothing is true, throws an error otherwise (see the second method below)
    nothing
end

function inference_check_itertype(keyword::Symbol, ::T) where {T}
    error("""
          Keyword argument `$(keyword)` expects either `Tuple` or `Vector` as an input, but a value of type `$(T)` has been used.
          If you specify a `Tuple` with a single entry - make sure you put a trailing comma at then end, e.g. `(something, )`. 
          Note: Julia's parser interprets `(something)` and (something, ) differently. 
              The first expression simply ignores parenthesis around `something`. 
              The second expression defines `Tuple`with `something` as a first (and the last) entry.
          """)
end

function infer_check_dicttype(::Symbol, ::Union{Nothing, NamedTuple, Dict, GraphPPL.VarDict})
    # This function check is the second argument is of type `Nothing`, `NamedTuple`, `Dict` or `VarDict`. 
    # Does nothing is true, throws an error otherwise (see the second method below)
    nothing
end

function infer_check_dicttype(keyword::Symbol, ::T) where {T}
    error("""
          Keyword argument `$(keyword)` expects either `Dict` or `NamedTuple` as an input, but a value of type `$(T)` has been used.
          If you specify a `NamedTuple` with a single entry - make sure you put a trailing comma at then end, e.g. `(x = something, )`. 
          Note: Julia's parser interprets `(x = something)` and (x = something, ) differently. 
              The first expression defines (or **overwrites!**) the local/global variable named `x` with `something` as a content. 
              The second expression defines `NamedTuple` with `x` as a key and `something` as a value.
          """)
end

inference_check_dataismissing(d) = (ismissing(d) || any(ismissing, d))

# Return NamedTuple for predictions
inference_fill_predictions(s::Symbol, d::AbstractArray) = NamedTuple{Tuple([s])}([repeat([missing], length(d))])
inference_fill_predictions(s::Symbol, d::DataVariable) = NamedTuple{Tuple([s])}([missing])

inference_invoke_callback(callbacks::T, name, args...) where {T} = _inference_invoke_callback(inference_get_callback(callbacks, name), args...)
inference_invoke_callback(::Nothing, name, args...) = nothing

_inference_invoke_callback(callback::T, args...) where {T} = callback(args...)
_inference_invoke_callback(::Nothing, args...) = nothing

inference_get_callback(callbacks, name) = get(() -> nothing, callbacks, name)
inference_get_callback(::Nothing, name) = nothing

unwrap_free_energy_option(option::Bool)                      = (option, Real)
unwrap_free_energy_option(option::Type{T}) where {T <: Real} = (true, T)

function available_callbacks end
function available_events end

function check_available_callbacks(warn, callbacks, ::Val{AvailableCallbacks}) where {AvailableCallbacks}
    if warn && !isnothing(callbacks)
        for key in keys(callbacks)
            if warn && key ∉ AvailableCallbacks
                @warn "Unknown callback specification: $(key). Available callbacks: $(AvailableCallbacks). Set `warn = false` to supress this warning."
            end
        end
    end
end

function check_available_events(warn, events::Nothing, ::Val{AvailableEvents}) where {AvailableEvents}
    # If `events` is nothing, we don't need to check anything
    return nothing
end

function check_available_events(warn, events::Val{Events}, ::Val{AvailableEvents}) where {Events, AvailableEvents}
    if warn && !isnothing(events)
        for key in Events
            if key ∉ AvailableEvents
                @warn "Unknown event type: $(key). Available events: $(AvailableEvents). Set `warn = false` to supress this warning."
            end
        end
    end
end

include("batch.jl")
include("autoupdates.jl")
include("streaming.jl")

"""
    infer(
        model; 
        data = nothing,
        datastream = nothing,
        autoupdates = nothing,
        initialization = nothing,
        constraints = nothing,
        meta = nothing,
        options = nothing,
        returnvars = nothing, 
        predictvars = nothing, 
        historyvars = nothing,
        keephistory = nothing,
        iterations = nothing,
        free_energy = false,
        free_energy_diagnostics = DefaultObjectiveDiagnosticChecks,
        showprogress = false,
        callbacks = nothing,
        addons = nothing,
        postprocess = DefaultPostprocess(),
        warn = true,
        events = nothing,
        uselock = false,
        autostart = true,
        catch_exception = false
    )

This function provides a generic way to perform probabilistic inference for batch/static and streamline/online scenarios.
Returns either an [`InferenceResult`](@ref) (batch setting) or [`RxInferenceEngine`](@ref) (streamline setting) based on the parameters used.

## Arguments

Check the official documentation for more information about some of the arguments. 

- `model`: specifies a model generator, required
- `data`: `NamedTuple` or `Dict` with data, required (or `datastream` or `predictvars`)
- `datastream`: A stream of `NamedTuple` with data, required (or `data`)
- `autoupdates = nothing`: auto-updates specification, required for streamline inference, see [`@autoupdates`](@ref)
- `initialization = nothing`: initialization specification object, optional, see [`@initialization`](@ref)
- `constraints = nothing`: constraints specification object, optional, see `@constraints`
- `meta  = nothing`: meta specification object, optional, may be required for some models, see `@meta`
- `options = nothing`: model creation options, optional, see [`ReactiveMPInferenceOptions`](@ref)
- `returnvars = nothing`: return structure info, optional, defaults to return everything at each iteration
- `predictvars = nothing`: return structure info, optional (exclusive for batch inference)
- `historyvars = nothing`: history structure info, optional, defaults to no history (exclusive for streamline inference)
- `keephistory = nothing`: history buffer size, defaults to empty buffer (exclusive for streamline inference)
- `iterations = nothing`: number of iterations, optional, defaults to `nothing`, the inference engine does not distinguish between variational message passing or Loopy belief propagation or expectation propagation iterations
- `free_energy = false`: compute the Bethe free energy, optional, defaults to false. Can be passed a floating point type, e.g. `Float64`, for better efficiency, but disables automatic differentiation packages, such as ForwardDiff.jl
- `free_energy_diagnostics = DefaultObjectiveDiagnosticChecks`: free energy diagnostic checks, optional, by default checks for possible `NaN`s and `Inf`s. `nothing` disables all checks.
- `showprogress = false`: show progress module, optional, defaults to false (exclusive for batch inference)
- `catch_exception`  specifies whether exceptions during the inference procedure should be caught, optional, defaults to false (exclusive for batch inference)
- `callbacks = nothing`: inference cycle callbacks, optional
- `addons = nothing`: inject and send extra computation information along messages
- `postprocess = DefaultPostprocess()`: inference results postprocessing step, optional
- `events = nothing`: inference cycle events, optional (exclusive for streamline inference)
- `uselock = false`: specifies either to use the lock structure for the inference or not, if set to true uses `Base.Threads.SpinLock`. Accepts custom `AbstractLock`. (exclusive for streamline inference)
- `autostart = true`: specifies whether to call `RxInfer.start` on the created engine automatically or not (exclusive for streamline inference)
- `warn = true`: enables/disables warnings

"""
function infer(;
    model = nothing,
    data = nothing,
    datastream = nothing, # streamline specific
    autoupdates = nothing, # streamline specific
    initialization = nothing,
    initmessages = nothing, # removed, the error is thrown below for easier migration
    initmarginals = nothing, # removed, the error is thrown below for easier migration
    constraints = nothing,
    meta = nothing,
    options = nothing,
    returnvars = nothing,
    predictvars = nothing, # batch specific
    historyvars = nothing, # streamline specific
    keephistory = nothing, # streamline specific
    iterations = nothing,
    free_energy = false,
    free_energy_diagnostics = DefaultObjectiveDiagnosticChecks,
    showprogress = false, # batch specific
    catch_exception = false, # batch specific
    callbacks = nothing,
    addons = nothing,
    postprocess = DefaultPostprocess(),
    events = nothing, # streamline specific
    uselock = false, # streamline  specific
    autostart = true, # streamline specific
    warn = true
)
    if isnothing(model)
        error("The `model` keyword argument is required for the `infer` function.")
    elseif !isa(model, GraphPPL.ModelGenerator)
        error("The `model` keyword argument must be of type `GraphPPL.ModelGenerator`.")
    elseif !isnothing(data) && !isnothing(datastream)
        error("""`data` and `datastream` keyword arguments cannot be used together. """)
    elseif isnothing(data) && isnothing(predictvars) && isnothing(datastream)
        error("""One of the keyword arguments `data` or `predictvars` or `datastream` must be specified""")
    elseif !isnothing(initmessages) && !isnothing(initmarginals)
        error(
            """`initmessages` and `initmarginals` keyword arguments have been deprecated and removed. Use the `@initialization` macro and the `initialization` keyword instead."""
        )
    end

    infer_check_dicttype(:callbacks, callbacks)
    infer_check_dicttype(:data, data)

    if isnothing(autoupdates)
        check_available_callbacks(warn, callbacks, available_callbacks(batch_inference))
        check_available_events(warn, events, available_events(batch_inference))
        batch_inference(
            model = model,
            data = data,
            initialization = initialization,
            constraints = constraints,
            meta = meta,
            options = options,
            returnvars = returnvars,
            predictvars = predictvars,
            iterations = iterations,
            free_energy = free_energy,
            free_energy_diagnostics = free_energy_diagnostics,
            showprogress = showprogress,
            callbacks = callbacks,
            addons = addons,
            postprocess = postprocess,
            warn = warn,
            catch_exception = catch_exception
        )
    else
        check_available_callbacks(warn, callbacks, available_callbacks(streaming_inference))
        check_available_events(warn, events, available_events(streaming_inference))
        streaming_inference(
            model = model,
            data = data,
            datastream = datastream,
            autoupdates = autoupdates,
            initialization = initialization,
            constraints = constraints,
            meta = meta,
            options = options,
            returnvars = returnvars,
            historyvars = historyvars,
            keephistory = keephistory,
            iterations = iterations,
            free_energy = free_energy,
            free_energy_diagnostics = free_energy_diagnostics,
            autostart = autostart,
            callbacks = callbacks,
            addons = addons,
            postprocess = postprocess,
            warn = warn,
            events = events,
            uselock = uselock
        )
    end
end
