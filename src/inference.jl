export KeepEach, KeepLast
export inference, InferenceResult
export rxinference, @autoupdates, RxInferenceEngine, RxInferenceEvent

import DataStructures: CircularBuffer

using MacroTools # for `@autoupdates`

import ReactiveMP: israndom, isdata, isconst, isproxy, isanonymous
import ReactiveMP: InfCountingReal

import ProgressMeter

obtain_marginal(variable::AbstractVariable, strategy = SkipInitial())                   = getmarginal(variable, strategy)
obtain_marginal(variables::AbstractArray{<:AbstractVariable}, strategy = SkipInitial()) = getmarginals(variables, strategy)

assign_marginal!(variables::AbstractArray{<:AbstractVariable}, marginals) = setmarginals!(variables, marginals)
assign_marginal!(variable::AbstractVariable, marginal)                    = setmarginal!(variable, marginal)

assign_message!(variables::AbstractArray{<:AbstractVariable}, messages) = setmessages!(variables, messages)
assign_message!(variable::AbstractVariable, message)                    = setmessage!(variable, message)

struct KeepEach end
struct KeepLast end

make_actor(::RandomVariable, ::KeepEach)                       = keep(Marginal)
make_actor(::Array{<:RandomVariable, N}, ::KeepEach) where {N} = keep(Array{Marginal, N})
make_actor(x::AbstractArray{<:RandomVariable}, ::KeepEach)     = keep(typeof(similar(x, Marginal)))

make_actor(::RandomVariable, ::KeepEach, capacity::Integer)                      = circularkeep(Marginal, capacity)
make_actor(::Array{<:RandomVariable, N}, ::KeepEach, capcity::Integer) where {N} = circularkeep(Array{Marginal, N}, capacity)
make_actor(x::AbstractArray{<:RandomVariable}, ::KeepEach, capacity::Integer)    = circularkeep(typeof(similar(x, Marginal)), capacity)

make_actor(::RandomVariable, ::KeepLast)                   = storage(Marginal)
make_actor(x::AbstractArray{<:RandomVariable}, ::KeepLast) = buffer(Marginal, size(x))

make_actor(::RandomVariable, ::KeepLast, capacity::Integer)                   = storage(Marginal)
make_actor(x::AbstractArray{<:RandomVariable}, ::KeepLast, capacity::Integer) = buffer(Marginal, size(x))

## Inference ensure update

import Rocket: Actor, on_next!, on_error!, on_complete!

# We can use `MarginalHasBeenUpdated` both as an actor in within the `ensure_update` operator
mutable struct MarginalHasBeenUpdated <: Actor{Any}
    updated::Bool
end

__unset_updated!(updated::MarginalHasBeenUpdated) = updated.updated = false
__set_updated!(updated::MarginalHasBeenUpdated)   = updated.updated = true

Rocket.on_next!(updated::MarginalHasBeenUpdated, anything) = __set_updated!(updated)
Rocket.on_error!(updated::MarginalHasBeenUpdated, err)     = begin end
Rocket.on_complete!(updated::MarginalHasBeenUpdated)       = begin end

# This creates a `tap` operator that will set the `updated` flag to true. 
# Later on we check flags and `unset!` them after the `update!` procedure
ensure_update(model::FactorGraphModel, callback, variable_name::Symbol, updated::MarginalHasBeenUpdated) =
    tap() do update
        __set_updated!(updated)
        callback(model, variable_name, update)
    end

ensure_update(model::FactorGraphModel, ::Nothing, variable_name::Symbol, updated::MarginalHasBeenUpdated) =
    tap() do _
        __set_updated!(updated) # If `callback` is nothing we simply set updated flag
    end

function __check_and_unset_updated!(updates)
    if all((v) -> v.updated, values(updates))
        foreach(__unset_updated!, values(updates))
    else
        not_updated = filter((pair) -> !last(pair).updated, updates)
        names = join(keys(not_updated), ", ")
        error("""
              Variables [ $(names) ] have not been updated after an update event. 
              Therefore, make sure to initialize all required marginals and messages. See `initmarginals` and `initmessages` keyword arguments for the inference function. 
              """)
    end
end

## Extra error handling

__inference_process_error(error) = rethrow(error)

function __inference_process_error(err::StackOverflowError)
    @error """
    Stack overflow error occurred during the inference procedure. 
    The inference engine may execute message update rules recursively, hence, the model graph size might be causing this error. 
    To resolve this issue, try using `limit_stack_depth` inference option for model creation. See `?inference` documentation for more details.
    The `limit_stack_depth` option does not help against over stack overflow errors that might hapenning outside of the model creation or message update rules execution.
    """
    rethrow(err) # Shows the original stack trace
end

"""
    __inference_check_itertype(label, container)

This function check is the second argument is of type `Nothing`, `Tuple` or `Vector`. Throws an error otherwise.
"""
function __inference_check_itertype end

__inference_check_itertype(::Symbol, ::Union{Nothing, Tuple, Vector}) = nothing

function __inference_check_itertype(keyword::Symbol, ::T) where {T}
    error("""
          Keyword argument `$(keyword)` expects either `Tuple` or `Vector` as an input, but a value of type `$(T)` has been used.
          If you specify a `Tuple` with a single entry - make sure you put a trailing comma at then end, e.g. `(something, )`. 
          Note: Julia's parser interprets `(something)` and (something, ) differently. 
              The first expression simply ignores parenthesis around `something`. 
              The second expression defines `Tuple`with `something` as a first (and the last) entry.
          """)
end

"""
    __inference_check_dicttype(label, container)

This function check is the second argument is of type `Nothing`, `NamedTuple` or `Dict`. Throws an error otherwise.
"""
function __inference_check_dicttype end

__inference_check_dicttype(::Symbol, ::Union{Nothing, NamedTuple, Dict}) = nothing

function __inference_check_dicttype(keyword::Symbol, ::T) where {T}
    error("""
          Keyword argument `$(keyword)` expects either `Dict` or `NamedTuple` as an input, but a value of type `$(T)` has been used.
          If you specify a `NamedTuple` with a single entry - make sure you put a trailing comma at then end, e.g. `(x = something, )`. 
          Note: Julia's parser interprets `(x = something)` and (x = something, ) differently. 
              The first expression defines (or **overwrites!**) the local/global variable named `x` with `something` as a content. 
              The second expression defines `NamedTuple` with `x` as a key and `something` as a value.
          """)
end

##

"""
    InferenceResult

This structure is used as a return value from the [`inference`](@ref) function. 

# Public Fields

- `posteriors`: `Dict` or `NamedTuple` of 'random variable' - 'posterior' pairs. See the `returnvars` argument for [`inference`](@ref).
- `free_energy`: (optional) An array of Bethe Free Energy values per VMP iteration. See the `free_energy` argument for [`inference`](@ref).
- `model`: `FactorGraphModel` object reference.
- `returnval`: Return value from executed `@model`.

See also: [`inference`](@ref)
"""
struct InferenceResult{P, F, M, R}
    posteriors  :: P
    free_energy :: F
    model       :: M
    returnval   :: R
end

Base.iterate(results::InferenceResult)      = iterate((getfield(results, :posteriors), getfield(results, :free_energy), getfield(results, :model), getfield(results, :returnval)))
Base.iterate(results::InferenceResult, any) = iterate((getfield(results, :posteriors), getfield(results, :free_energy), getfield(results, :model), getfield(results, :returnval)), any)

function Base.show(io::IO, result::InferenceResult)
    print(io, "Inference results:\n")

    lcolumnlen = 18 # Defines the padding for the "left" column of the output

    print(io, rpad("  Posteriors", lcolumnlen), " | ")
    print(io, "available for (")
    join(io, keys(getfield(result, :posteriors)), ", ")
    print(io, ")\n")

    if !isnothing(getfield(result, :free_energy))
        print(io, rpad("  Free Energy:", lcolumnlen), " | ")
        print(IOContext(io, :compact => true, :limit => true, :displaysize => (1, 80)), result.free_energy)
        print(io, "\n")
    end
end

function Base.getproperty(result::InferenceResult, property::Symbol)
    if property === :free_energy && getfield(result, :free_energy) === nothing
        error("""
              Bethe Free Energy has not been computed. 
              Use `free_energy = true` keyword argument for the `inference` function to compute Bethe Free Energy values.
              """)
    else
        return getfield(result, property)
    end
    return getfield(result, property)
end

__inference_invoke_callback(callback, args...)  = callback(args...)
__inference_invoke_callback(::Nothing, args...) = begin end

inference_invoke_callback(callbacks, name, args...) = __inference_invoke_callback(inference_get_callback(callbacks, name), args...)
inference_invoke_callback(::Nothing, name, args...) = begin end

inference_get_callback(callbacks, name) = get(() -> nothing, callbacks, name)
inference_get_callback(::Nothing, name) = nothing

unwrap_free_energy_option(option::Bool)                      = (option, Real, InfCountingReal)
unwrap_free_energy_option(option::Type{T}) where {T <: Real} = (true, T, InfCountingReal{T})

"""
    inference(
        model; 
        data,
        initmarginals           = nothing,
        initmessages            = nothing,
        constraints             = nothing,
        meta                    = nothing,
        options                 = nothing,
        returnvars              = nothing, 
        iterations              = nothing,
        free_energy             = false,
        free_energy_diagnostics = BetheFreeEnergyDefaultChecks,
        showprogress            = false,
        callbacks               = nothing,
    )

This function provides a generic way to perform probabilistic inference in RxInfer.jl. Returns `InferenceResult`.

## Arguments

For more information about some of the arguments, please check below.

- `model`: specifies a model generator, required
- `data`: `NamedTuple` or `Dict` with data, required
- `initmarginals = nothing`: `NamedTuple` or `Dict` with initial marginals, optional
- `initmessages = nothing`: `NamedTuple` or `Dict` with initial messages, optional
- `constraints = nothing`: constraints specification object, optional, see `@constraints`
- `meta  = nothing`: meta specification object, optional, may be required for some models, see `@meta`
- `options = nothing`: model creation options, optional, see `ModelInferenceOptions`
- `returnvars = nothing`: return structure info, optional, defaults to return everything at each iteration, see below for more information
- `iterations = nothing`: number of iterations, optional, defaults to `nothing`, the inference engine does not distinguish between variational message passing or Loopy belief propagation or expectation propagation iterations, see below for more information
- `free_energy = false`: compute the Bethe free energy, optional, defaults to false. Can be passed a floating point type, e.g. `Float64`, for better efficiency, but disables automatic differentiation packages, such as ForwardDiff.jl
- `free_energy_diagnostics = BetheFreeEnergyDefaultChecks`: free energy diagnostic checks, optional, by default checks for possible `NaN`s and `Inf`s. `nothing` disables all checks.
- `showprogress = false`: show progress module, optional, defaults to false
- `callbacks = nothing`: inference cycle callbacks, optional, see below for more info
- `warn = true`: enables/disables warnings

## Note on NamedTuples

When passing `NamedTuple` as a value for some argument, make sure you use a trailing comma for `NamedTuple`s with a single entry. The reason is that Julia treats `returnvars = (x = KeepLast())` and 
`returnvars = (x = KeepLast(), )` expressions differently. First expression creates (or **overwrites!**) new local/global variable named `x` with contents `KeepLast()`. The second expression (note traling comma)
creates `NamedTuple` with `x` as a key and `KeepLast()` as a value assigned for this key.

## Extended information about some of the arguments

- ### `model`

The `model` argument accepts a `ModelGenerator` as its input. The easiest way to create the `ModelGenerator` is to use the `@model` macro. 
For example:

```julia
@model function coin_toss(some_argument, some_keyword_argument = 3)
   ...
end

result = inference(
    model = coin_toss(some_argument; some_keyword_argument = 3)
)
```

**Note**: The `model` keyword argument does not accept a `FactorGraphModel` instance as a value, as it needs to inject `constraints` and `meta` during the inference procedure.

- ### `initmarginals`

In general for variational message passing every marginal distribution in a model needs to be pre-initialised. In practice, however, for many models it is sufficient enough to initialise only a small subset of variables in the model.

- ### `initmessages`

Loopy belief propagation may need some messages in a model to be pre-initialised.

- ### `options`

- `limit_stack_depth`: limits the stack depth for computing messages, helps with `StackOverflowError` for some huge models, but reduces the performance of inference backend. Accepts integer as an argument that specifies the maximum number of recursive depth. Lower is better for stack overflow error, but worse for performance.
- `pipeline`: changes the default pipeline for each factor node in the graph
- `global_reactive_scheduler`: changes the scheduler of reactive streams, see Rocket.jl for more info, defaults to no scheduler

- ### `returnvars`

`returnvars` specifies the variables of interests and the amount of information to return about their posterior updates. 

`returnvars` accepts a `NamedTuple` or `Dict` or return var specification. There are two specifications:
- `KeepLast`: saves the last update for a variable, ignoring any intermediate results during iterations
- `KeepEach`: saves all updates for a variable for all iterations

Note: if `iterations` are specified as a number, the `inference` function tracks and returns every update for each iteration for every random variable in the model (equivalent to `KeepEach()`).
If number of iterations is set to `nothing`, the `inference` function saves the 'last' (and the only one) update for every random variable in the model (equivalent to `KeepLast()`). 
Use `iterations = 1` to force `KeepEach()` setting when number of iterations is equal to `1` or set `returnvars = KeepEach()` manually.

Example: 

```julia
result = inference(
    ...,
    returnvars = (
        x = KeepLast(),
        τ = KeepEach()
    )
)
```

It is also possible to set iether `returnvars = KeepLast()` or `returnvars = KeepEach()` that acts as an alias and sets the given option for __all__ random variables in the model.

# Example: 

```julia
result = inference(
    ...,
    returnvars = KeepLast()
)
```

- ### `iterations`

Specifies the number of variational (or loopy BP) iterations. By default set to `nothing`, which is equivalent of doing 1 iteration. 

- ### `free_energy` 

This setting specifies whenever the `inference` function should return Bethe Free Energy (BFE) values. 
Note, however, that it may be not possible to compute BFE values for every model. 

Additionally, the argument may accept a floating point type, instead of a `Bool` value. Using his option, e.g.`Float64`, improves performance of Bethe Free Energy computation, but restricts using automatic differentiation packages.

- ### `free_energy_diagnostics`

This settings specifies either a single or a tuple of diagnostic checks for Bethe Free Energy values stream. By default checks for `NaN`s and `Inf`s. See also [`BetheFreeEnergyCheckNaNs`](@ref) and [`BetheFreeEnergyCheckInfs`](@ref).
Pass `nothing` to disable any checks.

- ### `callbacks`

The inference function has its own lifecycle. The user is free to provide some (or none) of the callbacks to inject some extra logging or other procedures in the inference function, e.g.

```julia
result = inference(
    ...,
    callbacks = (
        on_marginal_update = (model, name, update) -> println("\$(name) has been updated: \$(update)"),
        after_inference    = (args...) -> println("Inference has been completed")
    )
)
```

The `callbacks` keyword argument accepts a named-tuple of 'name = callback' pairs. 
The list of all possible callbacks and their arguments is present below:

- `on_marginal_update`:    args: (model::FactorGraphModel, name::Symbol, update)
- `before_model_creation`: args: ()
- `after_model_creation`:  args: (model::FactorGraphModel, returnval)
- `before_inference`:      args: (model::FactorGraphModel)
- `before_iteration`:      args: (model::FactorGraphModel, iteration::Int)
- `before_data_update`:    args: (model::FactorGraphModel, data)
- `after_data_update`:     args: (model::FactorGraphModel, data)
- `after_iteration`:       args: (model::FactorGraphModel, iteration::Int)
- `after_inference`:       args: (model::FactorGraphModel)

See also: [`InferenceResult`](@ref)
"""
function inference(;
    # `model`: specifies a model generator, with the help of the `Model` function
    model::ModelGenerator,
    # NamedTuple or Dict with data, required
    data,
    # NamedTuple or Dict with initial marginals, optional, defaults to empty
    initmarginals = nothing,
    # NamedTuple or Dict with initial messages, optional, defaults to empty
    initmessages = nothing,  # optional
    # Constraints specification object
    constraints = nothing,
    # Meta specification object
    meta = nothing,
    # Model creation options
    options = nothing,
    # Return structure info, optional, defaults to return everything at each iteration
    returnvars = nothing,
    # Number of iterations, defaults to 1, we do not distinguish between VMP or Loopy belief or EP iterations
    iterations = nothing,
    # Do we compute FE, optional, defaults to false 
    # Can be passed a floating point type, e.g. `Float64`, for better efficiency, but disables automatic differentiation packages, such as ForwardDiff.jl
    free_energy = false,
    # Default BFE stream checks
    free_energy_diagnostics = BetheFreeEnergyDefaultChecks,
    # Show progress module, optional, defaults to false
    showprogress = false,
    # Inference cycle callbacks
    callbacks = nothing,
    # warn, optional, defaults to true
    warn = true
)
    __inference_check_dicttype(:data, data)
    __inference_check_dicttype(:initmarginals, initmarginals)
    __inference_check_dicttype(:initmessages, initmessages)
    __inference_check_dicttype(:callbacks, callbacks)

    # Check for available callbacks
    if warn && !isnothing(callbacks)
        for key in keys(callbacks)
            if key ∉ (
                :on_marginal_update,
                :before_model_creation,
                :after_model_creation,
                :before_inference,
                :before_iteration,
                :before_data_update,
                :after_data_update,
                :after_iteration,
                :after_inference
            )
                @warn "Unknown callback specification: $(key). Available callbacks: on_marginal_update, before_model_creation, after_model_creation, before_inference, before_iteration, before_data_update, after_data_update, after_iteration, after_inference. Set `warn = false` to supress this warning."
            end
        end
    end

    _options = convert(ModelInferenceOptions, options)

    inference_invoke_callback(callbacks, :before_model_creation)
    fmodel, freturval = create_model(model, constraints = constraints, meta = meta, options = _options)
    inference_invoke_callback(callbacks, :after_model_creation, fmodel, freturval)
    vardict = getvardict(fmodel)

    # First what we do - we check if `returnvars` is nothing or one of the two possible values: `KeepEach` and `KeepLast`. 
    # If so, we replace it with either `KeepEach` or `KeepLast` for each random and not-proxied variable in a model
    if returnvars === nothing || returnvars === KeepEach() || returnvars === KeepLast()
        # Checks if the first argument is `nothing`, in which case returns the second argument
        returnoption = something(returnvars, iterations isa Number ? KeepEach() : KeepLast())
        returnvars   = Dict(variable => returnoption for (variable, value) in pairs(vardict) if (israndom(value) && !isanonymous(value)))
    end

    __inference_check_dicttype(:returnvars, returnvars)

    # Use `__check_has_randomvar` to filter out unknown or non-random variables in the `returnvar` specification
    __check_has_randomvar(vardict, variable) = begin
        haskey_check   = haskey(vardict, variable)
        israndom_check = haskey_check ? israndom(vardict[variable]) : false
        if warn && !haskey_check
            @warn "`returnvars` object has `$(variable)` specification, but model has no variable named `$(variable)`. The `$(variable)` specification is ignored. Use `warn = false` to suppress this warning."
        elseif warn && haskey_check && !israndom_check
            @warn "`returnvars` object has `$(variable)` specification, but model has no **random** variable named `$(variable)`. The `$(variable)` specification is ignored. Use `warn = false` to suppress this warning."
        end
        return haskey_check && israndom_check
    end

    # Second, for each random variable entry we create an actor
    actors = Dict(variable => make_actor(vardict[variable], value) for (variable, value) in pairs(returnvars) if __check_has_randomvar(vardict, variable))

    # At third, for each random variable entry we create a boolean flag to track their updates
    updates = Dict(variable => MarginalHasBeenUpdated(false) for (variable, _) in pairs(actors))

    _iterations = something(iterations, 1)
    _iterations isa Integer || error("`iterations` argument must be of type Integer or `nothing`")
    _iterations > 0 || error("`iterations` arguments must be greater than zero")

    try
        on_marginal_update = inference_get_callback(callbacks, :on_marginal_update)
        subscriptions      = Dict(variable => subscribe!(obtain_marginal(vardict[variable]) |> ensure_update(fmodel, on_marginal_update, variable, updates[variable]), actor) for (variable, actor) in pairs(actors))

        fe_actor        = nothing
        fe_subscription = VoidTeardown()

        is_free_energy, S, T = unwrap_free_energy_option(free_energy)

        if is_free_energy
            fe_actor        = ScoreActor(S, _iterations, 1)
            fe_objective    = BetheFreeEnergy(BetheFreeEnergyDefaultMarginalSkipStrategy, AsapScheduler(), free_energy_diagnostics)
            fe_subscription = subscribe!(score(fmodel, T, fe_objective), fe_actor)
        end

        if !isnothing(initmarginals)
            for (variable, initvalue) in pairs(initmarginals)
                if haskey(vardict, variable)
                    assign_marginal!(vardict[variable], initvalue)
                elseif warn
                    @warn "`initmarginals` object has `$(variable)` specification, but model has no variable named `$(variable)`. Use `warn = false` to suppress this warning."
                end
            end
        end

        if !isnothing(initmessages)
            for (variable, initvalue) in pairs(initmessages)
                if haskey(vardict, variable)
                    assign_message!(vardict[variable], initvalue)
                elseif warn
                    @warn "`initmessages` object has `$(variable)` specification, but model has no variable named `$(variable)`. Use `warn = false` to suppress this warning."
                end
            end
        end

        if isnothing(data) || isempty(data)
            error("Data is empty. Make sure you used `data` keyword argument with correct value.")
        else
            foreach(filter(pair -> isdata(last(pair)), pairs(vardict))) do pair
                varname = first(pair)
                haskey(data, varname) || error("Data entry `$(varname)` is missing in `data` argument. Double check `data = ($(varname) = ???, )`")
            end
        end

        inference_invoke_callback(callbacks, :before_inference, fmodel)

        fdata = filter(pairs(data)) do pair
            hk      = haskey(vardict, first(pair))
            is_data = hk ? isdata(vardict[first(pair)]) : false
            if warn && (!hk || !is_data)
                @warn "`data` object has `$(first(pair))` specification, but model has no data input named `$(first(pair))`. Use `warn = false` to suppress this warning."
            end
            return hk && is_data
        end

        p = showprogress ? ProgressMeter.Progress(_iterations) : nothing

        for iteration in 1:_iterations
            inference_invoke_callback(callbacks, :before_iteration, fmodel, iteration)
            inference_invoke_callback(callbacks, :before_data_update, fmodel, data)
            for (key, value) in fdata
                update!(vardict[key], value)
            end
            inference_invoke_callback(callbacks, :after_data_update, fmodel, data)

            # Check that all requested marginals have been updated and unset the `updated` flag
            # Throws an error if some were not update
            __check_and_unset_updated!(updates)

            if !isnothing(p)
                ProgressMeter.next!(p)
            end
            inference_invoke_callback(callbacks, :after_iteration, fmodel, iteration)
        end

        for (_, subscription) in pairs(subscriptions)
            unsubscribe!(subscription)
        end

        if !isnothing(fe_actor)
            release!(fe_actor)
        end

        unsubscribe!(fe_subscription)

        posterior_values = Dict(variable => ReactiveMP.getdata(getvalues(actor)) for (variable, actor) in pairs(actors))
        fe_values        = !isnothing(fe_actor) ? score_snapshot_iterations(fe_actor) : nothing

        inference_invoke_callback(callbacks, :after_inference, fmodel)

        return InferenceResult(posterior_values, fe_values, fmodel, freturval)
    catch error
        __inference_process_error(error)
    end
end

## ------------------------------------------------------------------------ ##

struct FromMarginalAutoUpdate end
struct FromMessageAutoUpdate end

import Base: string

Base.string(::FromMarginalAutoUpdate) = "q"
Base.string(::FromMessageAutoUpdate) = "μ"

import Base: fetch

# TODO for arrays
Base.fetch(::FromMarginalAutoUpdate, variable::Union{DataVariable, RandomVariable}) = ReactiveMP.getmarginal(variable, IncludeAll())

Base.fetch(::FromMessageAutoUpdate, variable::RandomVariable) = ReactiveMP.messagein(variable, 1) # Here we assume that predictive message has index `1`
Base.fetch(::FromMessageAutoUpdate, variable::DataVariable)   = error("`FromMessageAutoUpdate` fetch strategy is not implemented for `DataVariable`")

struct RxInferenceAutoUpdateSpecification{N, F, C}
    labels   :: NTuple{N, Symbol}
    from     :: F
    callback :: C
    variable :: Symbol
end

function Base.show(io::IO, specification::RxInferenceAutoUpdateSpecification)
    print(io, join(specification.labels, ","), " = ", string(specification.callback), "(", string(specification.from), "(", specification.variable, "))")
end

function (specification::RxInferenceAutoUpdateSpecification)(model::FactorGraphModel)
    datavars = map(specification.labels) do label
        hasdatavar(model, label) || error("Autoupdate specification defines an update for `$(label)`, but the model has no datavar named `$(label)`")
        return model[label]
    end

    (hasrandomvar(model, specification.variable) || hasdatavar(model, specification.variable)) ||
        error("Autoupdate specification defines an update from `$(specification.variable)`, but the model has no randomvar/datavar named `$(specification.variable)`")

    variable = model[specification.variable]

    return RxInferenceAutoUpdate(datavars, specification.callback, fetch(specification.from, variable))
end

struct RxInferenceAutoUpdate{N, C, R}
    datavars :: N
    callback :: C
    recent   :: R
end

import Base: fetch

# TODO for arrays
Base.fetch(autoupdate::RxInferenceAutoUpdate)            = fetch(autoupdate, ReactiveMP.getdata(Rocket.getrecent(autoupdate.recent)))
Base.fetch(autoupdate::RxInferenceAutoUpdate, something) = zip(as_tuple(autoupdate.datavars), as_tuple(autoupdate.callback(something)))

"""
    @autoupdates

Creates the auto-updates specification for the `rxinference` function. In the online-streaming Bayesian inference procedure it is important to update your priors for the future 
states based on the new updated posteriors. The `@autoupdates` structure simplify such a specification. It accepts a single block of code where each line defines how to update 
the `datavar`'s in the probabilistic model specification. 

Each line of code in the auto-update specification defines `datavar`s, which need to be updated, on the left hand side of the equality expression and the update function on the right hand side of the expression.
The update function operates on posterior marginals in the form of the `q(symbol)` expression.

For example:

```julia
@autoupdates begin 
    x = f(q(z))
end
```

This structure specifies to automatically update `x = datavar(...)` as soon as the inference engine computes new posterior over `z` variable. It then applies the `f` function
to the new posterior and calls `update!(x, ...)` automatically. 

As an example consider the following model and auto-update specification:

```julia
@model function kalman_filter()
    x_current_mean = datavar(Float64)
    x_current_var  = datavar(Float64)

    x_current ~ Normal(mean = x_current_mean, var = x_current_var)

    x_next ~ Normal(mean = x_current, var = 1.0)

    y = datavar(Float64)
    y ~ Normal(mean = x_next, var = 1.0)
end
```

This model has two `datavar`s that represent our prior knowledge of the `x_current` state of the system. The `x_next` random variable represent the next state of the system that 
is connected to the observed variable `y`. The auto-update specification could look like:

```jldoctest
autoupdates = @autoupdates begin
    x_current_mean, x_current_var = mean_cov(q(x_next))
end;
```

This structure specifies to update our prior as soon as we have a new posterior `q(x_next)`. It then applies the `mean_cov` function on the updated posteriors and updates 
`datavar`s `x_current_mean` and `x_current_var` automatically.

See also: [`rxinference`](@ref)
"""
macro autoupdates(code)
    ((code isa Expr) && (code.head === :block)) || error("Autoupdate requires a block of code `begin ... end` as an input")

    specifications = []

    code = MacroTools.postwalk(code) do expression
        # We modify all expression of the form `... = callback(q(...))` or `... = callback(μ(...))`
        if @capture(expression, (lhs_ = callback_(rhs_)) | (lhs_ = callback_(rhs__)))
            if @capture(rhs, (q(variable_)) | (μ(variable_)))
                # First we check that `variable` is a plain Symbol
                (variable isa Symbol) || error("Variable in the expression `$(expression)` must be a plain name, but a complex expression `$(variable)` found.")
                # Next we extract `datavars` specification from the `lhs`                    
                datavars = if lhs isa Symbol
                    (lhs,)
                elseif lhs isa Expr && lhs.head === :tuple && all(arg -> arg isa Symbol, lhs.args)
                    Tuple(lhs.args)
                else
                    error("Left hand side of the expression `$(expression)` must be a single symbol or a tuple of symbols")
                end
                # Only two options are possible within this `if` block
                from = @capture(rhs, q(smth_)) ? :(RxInfer.FromMarginalAutoUpdate()) : :(RxInfer.FromMessageAutoUpdate())

                push!(specifications, :(RxInfer.RxInferenceAutoUpdateSpecification($(datavars...,), $from, $callback, $(QuoteNode(variable)))))

                return :(nothing)
            else
                error("Complex call expression `$(expression)` in the `@autoupdates` macro")
            end
        else
            return expression
        end
    end

    isempty(specifications) && error("`@autoupdates` did not find any auto-updates specifications. Check the documentation for more information.")

    output = quote
        begin
            $code

            ($(specifications...),)
        end
    end

    return esc(output)
end

## ------------------------------------------------------------------------ ##

"""
    RxInferenceEngine

The return value of the `rxinference` function. 

# Public fields
- `posteriors`: `Dict` or `NamedTuple` of 'random variable' - 'posterior stream' pairs. See the `returnvars` argument for the [`rxinference`](@ref).
- `free_energy`: (optional) A stream of Bethe Free Energy values per VMP iteration. See the `free_energy` argument for the [`rxinference`](@ref).
- `history`: (optional) Saves history of previous marginal updates. See the `historyvars` and `keephistory` arguments for the [`rxinference`](@ref).
- `free_energy_history`: (optional) Free energy history, average over variational iterations 
- `free_energy_raw_history`: (optional) Free energy history, returns returns computed values of all variational iterations for each data event (if available)
- `free_energy_final_only_history`: (optional) Free energy history, returns computed values of final variational iteration for each data event (if available)
- `events`: (optional) A stream of events send by the inference engine. See the `events` argument for the [`rxinference`](@ref).
- `model`: `FactorGraphModel` object reference.
- `returnval`: Return value from executed `@model`.

Use the `RxInfer.start(engine)` function to subscribe on the `data` source and start the inference procedure. Use `RxInfer.stop(engine)` to unsubscribe from the `data` source and stop the inference procedure. 
Note, that it is not always possible to start/stop the inference procedure.

See also: [`rxinference`](@ref), [`RxInferenceEvent`](@ref), [`RxInfer.start`](@ref), [`RxInfer.stop`](@ref)
"""
mutable struct RxInferenceEngine{T, D, L, V, P, H, S, U, A, FA, FH, FO, FS, I, M, N, X, E, J}
    datastream       :: D
    tickscheduler    :: L
    mainsubscription :: Teardown

    datavars   :: V
    posteriors :: P

    history::H
    historyactors::S
    historysubscriptions::Vector{Teardown}

    updateflags::U
    updatesubscriptions::Vector{Teardown}

    # auto updates
    autoupdates::A

    # free energy related
    fe_actor        :: FA
    fe_scheduler    :: FH
    fe_objective    :: FO
    fe_source       :: FS
    fe_subscription :: Teardown

    # utility 
    iterations   :: I
    model        :: M
    returnval    :: N
    events       :: E
    is_running   :: Bool
    is_errored   :: Bool
    is_completed :: Bool
    error        :: Any
    ticklock     :: J

    RxInferenceEngine(
        ::Type{T},
        datastream::D,
        tickscheduler::L,
        datavars::V,
        posteriors::P,
        updateflags::U,
        history::H,
        historyactors::S,
        autoupdates::A,
        fe_actor::FA,
        fe_scheduler::FH,
        fe_objective::FO,
        fe_source::FS,
        iterations::I,
        model::M,
        returnval::N,
        enabledevents::Val{X},
        events::E,
        ticklock::J
    ) where {T, D, L, V, P, H, S, U, A, FA, FH, FO, FS, I, M, N, X, E, J} = begin
        return new{T, D, L, V, P, H, S, U, A, FA, FH, FO, FS, I, M, N, X, E, J}(
            datastream,
            tickscheduler,
            voidTeardown,
            datavars,
            posteriors,
            history,
            historyactors,
            Teardown[],
            updateflags,
            Teardown[],
            autoupdates,
            fe_actor,
            fe_scheduler,
            fe_objective,
            fe_source,
            voidTeardown,
            iterations,
            model,
            returnval,
            events,
            false,
            false,
            false,
            nothing,
            ticklock
        )
    end
end

function Base.show(io::IO, engine::RxInferenceEngine)
    print(io, "RxInferenceEngine:\n")

    lcolumnlen = 22 # Defines the padding for the "left" column of the output

    print(io, rpad("  Posteriors stream", lcolumnlen), " | ")
    print(io, "enabled for (")
    join(io, keys(getfield(engine, :posteriors)), ", ")
    print(io, ")\n")

    print(io, rpad("  Free Energy stream", lcolumnlen), " | ")
    if !isnothing(getfield(engine, :fe_source))
        print(io, "enabled\n")
    else
        print(io, "disabled\n")
    end

    print(io, rpad("  Posteriors history", lcolumnlen), " | ")
    print(io, "available for (")
    join(io, keys(getfield(engine, :historyactors)), ", ")
    print(io, ")\n")

    print(io, rpad("  Free Energy history", lcolumnlen), " | ")
    if !isnothing(getfield(engine, :fe_actor))
        print(io, "available\n")
    else
        print(io, "unavailable\n")
    end

    print(io, rpad("  Enabled events", lcolumnlen), " | ")
    print(io, "[ ", join(enabled_events(engine), ", "), " ]")
end

enabled_events(::RxInferenceEngine{T, D, L, V, P, H, S, U, A, FA, FH, FO, FS, I, M, N, X, E}) where {T, D, L, V, P, H, S, U, A, FA, FH, FO, FS, I, M, N, X, E} = X

function Base.getproperty(result::RxInferenceEngine, property::Symbol)
    if property === :enabled_events
        return enabled_events(result)
    elseif property === :free_energy
        !isnothing(getfield(result, :fe_source)) ||
            error("Bethe Free Energy stream has not been created. Use `free_energy = true` keyword argument for the `rxinference` function to compute Bethe Free Energy values.")
        return getfield(result, :fe_source)
    elseif property === :free_energy_history
        !isnothing(getfield(result, :fe_actor)) || error(
            "Bethe Free Energy history has not been computed. Use `free_energy = true` keyword argument for the `rxinference` function to compute Bethe Free Energy values together with the `keephistory` argument."
        )
        return score_snapshot_iterations(getfield(result, :fe_actor))
    elseif property === :free_energy_final_only_history
        !isnothing(getfield(result, :fe_actor)) || error(
            "Bethe Free Energy history has not been comptued. Use `free_energy = true` keyword argument for the `rxinference` function to compute Bethe Free Energy values together with the `keephistory` argument."
        )
        return score_snapshot_final(getfield(result, :fe_actor))
    elseif property === :free_energy_raw_history
        !isnothing(getfield(result, :fe_actor)) || error(
            "Bethe Free Energy history has not been comptued. Use `free_energy = true` keyword argument for the `rxinference` function to compute Bethe Free Energy values together with the `keephistory` argument."
        )
        return score_snapshot(getfield(result, :fe_actor))
    end
    return getfield(result, property)
end

"""
    start(engine::RxInferenceEngine)

Starts the `RxInferenceEngine` by subscribing to the data source, instantiating free energy (if enabled) and starting the event loop.
Use [`RxInfer.stop`](@ref) to stop the `RxInferenceEngine`. Note that it is not always possible to stop/restart the engine and this depends on the data source type.

See also: [`RxInfer.stop`](@ref)
"""
function start(engine::RxInferenceEngine{T}) where {T}
    rxexecutorlock(engine.ticklock) do
        if engine.is_completed || engine.is_errored
            @warn "The engine has been completed or errored. Cannot start an exhausted engine."
            return nothing
        end

        if engine.is_running
            @warn "The engine is already running. Cannot start a single engine twice."
            return nothing
        end

        _enabled_events = engine.enabled_events
        _events         = engine.events

        inference_invoke_event(Val(:before_start), Val(_enabled_events), _events, engine)

        _eventexecutor = RxInferenceEventExecutor(T, engine)
        _tickscheduler = engine.tickscheduler

        # This subscription tracks updates of all `posteriors`
        engine.updatesubscriptions = map(keys(engine.updateflags), values(engine.updateflags)) do name, updateflag
            return subscribe!(obtain_marginal(engine.model[name]), updateflag)
        end

        if !isnothing(engine.historyactors) && !isnothing(engine.history)
            engine.historysubscriptions = map(keys(engine.historyactors), values(engine.historyactors)) do name, actor
                return subscribe!(obtain_marginal(engine.model[name]), actor)
            end
        end

        if !isnothing(engine.fe_actor)
            engine.fe_subscription = subscribe!(engine.fe_source, engine.fe_actor)
        end

        release!(_tickscheduler)

        engine.is_running = true

        # After all preparations we finaly can `subscribe!` on the `datastream`
        engine.mainsubscription = subscribe!(engine.datastream, _eventexecutor)

        inference_invoke_event(Val(:after_start), Val(_enabled_events), _events, engine)
    end

    return nothing
end

"""
    stop(engine::RxInferenceEngine)

Stops the `RxInferenceEngine` by unsubscribing to the data source, free energy (if enabled) and stopping the event loop.
Use [`RxInfer.start`](@ref) to start the `RxInferenceEngine` again. Note that it is not always possible to stop/restart the engine and this depends on the data source type.

See also: [`RxInfer.start`](@ref)
"""
function stop(engine::RxInferenceEngine)
    rxexecutorlock(engine.ticklock) do
        if engine.is_completed || engine.is_errored
            @warn "The engine has been completed or errored. Cannot stop an exhausted engine."
            return nothing
        end

        if !engine.is_running
            @warn "The engine is not running. Cannot stop an idle engine."
            return nothing
        end

        _enabled_events = engine.enabled_events
        _events         = engine.events

        inference_invoke_event(Val(:before_stop), Val(_enabled_events), _events, engine)

        unsubscribe!(engine.fe_subscription)
        unsubscribe!(engine.historysubscriptions)
        unsubscribe!(engine.updatesubscriptions)
        unsubscribe!(engine.mainsubscription)

        engine.is_running = false

        inference_invoke_event(Val(:after_stop), Val(_enabled_events), _events, engine)
    end

    return nothing
end

import Rocket: Actor, on_next!, on_error!, on_complete!

struct RxInferenceEventExecutor{T, E} <: Actor{T}
    engine::E

    RxInferenceEventExecutor(::Type{T}, engine::E) where {T, E} = new{T, E}(engine)
end

Base.show(io::IO, ::RxInferenceEventExecutor)         = print(io, "RxInferenceEventExecutor")
Base.show(io::IO, ::Type{<:RxInferenceEventExecutor}) = print(io, "RxInferenceEventExecutor")

rxexecutorlock(fn::F, ::Nothing) where {F} = fn()
rxexecutorlock(fn::F, locker) where {F}    = lock(fn, locker)

function Rocket.on_next!(executor::RxInferenceEventExecutor{T}, event::T) where {T}
    # This is the `main` executor of the inference procedure
    # It listens new data and is supposed to run indefinitely

    # By default `_ticklock` is nothing, `executorlock` is defined such that it does not sync if `_ticklock` is nothing
    _ticklock = executor.engine.ticklock

    rxexecutorlock(_ticklock) do

        # `executor.engine` is defined as mutable 
        # we extract all variables before the loop so Julia does not extract them every time
        _tickscheduler  = executor.engine.tickscheduler
        _iterations     = executor.engine.iterations
        _model          = executor.engine.model
        _datavars       = executor.engine.datavars
        _autoupdates    = executor.engine.autoupdates
        _updateflags    = executor.engine.updateflags
        _history        = executor.engine.history
        _historyactors  = executor.engine.historyactors
        _fe_actor       = executor.engine.fe_actor
        _fe_scheduler   = executor.engine.fe_scheduler
        _enabled_events = executor.engine.enabled_events
        _events         = executor.engine.events

        inference_invoke_event(Val(:on_new_data), Val(_enabled_events), _events, _model, event)

        # Before we start our iterations we 'prefetch' recent values for autoupdates
        fupdates = map(fetch, _autoupdates)

        # This loop correspond to the different VMP iterations
        for iteration in 1:_iterations
            inference_invoke_event(Val(:before_iteration), Val(_enabled_events), _events, _model, iteration)

            # At first we update all our priors (auto updates) with the fixed values from the `redirectupdate` field
            inference_invoke_event(Val(:before_auto_update), Val(_enabled_events), _events, _model, iteration, fupdates)
            foreach(fupdates) do fupdate
                for (datavar, value) in fupdate
                    update!(datavar, value)
                end
            end
            inference_invoke_event(Val(:after_auto_update), Val(_enabled_events), _events, _model, iteration, fupdates)

            # At second we pass our observations
            inference_invoke_event(Val(:before_data_update), Val(_enabled_events), _events, _model, iteration, event)
            for (datavar, value) in zip(_datavars, values(event))
                update!(datavar, value)
            end
            inference_invoke_event(Val(:after_data_update), Val(_enabled_events), _events, _model, iteration, event)

            __check_and_unset_updated!(_updateflags)

            if !isnothing(_fe_scheduler)
                release!(_fe_scheduler)
            end

            inference_invoke_event(Val(:after_iteration), Val(_enabled_events), _events, _model, iteration)
        end

        # `release!` on `fe_actor` ensures that free energy sumed up between iterations correctly
        if !isnothing(_fe_actor)
            release!(_fe_actor)
        end

        if !isnothing(_history) && !isnothing(_historyactors)
            inference_invoke_event(Val(:before_history_save), Val(_enabled_events), _events, _model)
            for (name, actor) in pairs(_historyactors)
                push!(_history[name], getdata(getvalues(actor)))
            end
            inference_invoke_event(Val(:after_history_save), Val(_enabled_events), _events, _model)
        end

        # On this `release!` call we update our priors for the next step
        release!(_tickscheduler)

        inference_invoke_event(Val(:on_tick), Val(_enabled_events), _events, _model)
    end
end

function Rocket.on_error!(executor::RxInferenceEventExecutor, err)
    _engine         = executor.engine
    _model          = executor.engine.model
    _enabled_events = executor.engine.enabled_events
    _events         = executor.engine.events

    _engine.is_errored = true
    _engine.error      = err

    inference_invoke_event(Val(:on_error), Val(_enabled_events), _events, _model, err)

    __inference_process_error(err)
end

function Rocket.on_complete!(executor::RxInferenceEventExecutor)
    _engine         = executor.engine
    _model          = executor.engine.model
    _enabled_events = executor.engine.enabled_events
    _events         = executor.engine.events

    _engine.is_completed = true

    inference_invoke_event(Val(:on_complete), Val(_enabled_events), _events, _model)

    return nothing
end

## 

"""
    RxInferenceEvent{T, D}

The `RxInferenceEngine` sends events in a form of the `RxInferenceEvent` structure. `T` represents the type of an event, `D` represents the type of a data associated with the event.
The type of data depends on the type of an event, but usually represents a tuple, which can be unrolled automatically with the Julia's splitting syntax, e.g. `model, iteration = event`. 
See the documentation of the `rxinference` function for possible event types and their associated data types.

The events system itself uses the `Rocket.jl` library API. For example, one may create a custom event listener in the following way:


```jldoctest
using Rocket

struct MyEventListener <: Rocket.Actor{RxInferenceEvent}
    # ... extra fields
end

function Rocket.on_next!(listener::MyEventListener, event::RxInferenceEvent{ :after_iteration })
    model, iteration = event
    println("Iteration \$(iteration) has been finished.")
end

function Rocket.on_error!(listener::MyEventListener, err)
    # ...
end

function Rocket.on_complete!(listener::MyEventListener)
    # ...
end

```

and later on:

```julia
engine = rxinference(events = Val((:after_iteration, )), ...)

subscription = subscribe!(engine.events, MyEventListener(...))
```

See also: [`rxinference`](@ref), [`RxInferenceEngine`](@ref)
"""
struct RxInferenceEvent{T, D}
    data::D

    RxInferenceEvent(::Val{T}, data::D) where {T, D} = new{T, D}(data)
end

RxInfer.name(::RxInferenceEvent{T}) where {T} = T

Base.show(io::IO, ::RxInferenceEvent{T}) where {T} = print(io, "RxInferenceEvent(:", T, ")")

Base.iterate(event::RxInferenceEvent)        = iterate(event.data)
Base.iterate(event::RxInferenceEvent, state) = iterate(event.data, state)

function inference_invoke_event(::Val{Event}, ::Val{EnabledEvents}, events, args...) where {Event, EnabledEvents}
    # Here `E` must be a tuple of symbols
    if Event ∈ EnabledEvents
        next!(events, RxInferenceEvent(Val(Event), args))
    end
    return nothing
end

##

"""
    rxinference(
        model,
        data = nothing,
        datastream = nothing,
        initmarginals = nothing,
        initmessages = nothing,
        autoupdates = nothing,
        constraints = nothing,
        meta = nothing,
        options = nothing,
        returnvars = nothing,
        historyvars = nothing,
        keephistory = nothing,
        iterations = nothing,
        free_energy = false,
        free_energy_diagnostics = BetheFreeEnergyDefaultChecks,
        autostart = true,
        events = nothing,
        callbacks = nothing,
        uselock = false,
        warn = true
    )

This function provides a generic way to perform probabilistic inference in RxInfer.jl. Returns `RxInferenceEngine`.

## Arguments

For more information about some of the arguments, please check below.

- `model`: specifies a model generator, required
- `data`: `NamedTuple` or `Dict` with data, required
- `initmarginals = nothing`: `NamedTuple` or `Dict` with initial marginals, optional
- `initmessages = nothing`: `NamedTuple` or `Dict` with initial messages, optional
- `autoupdates = nothing`: auto-updates specification, required for many models, see `@autoupdates`
- `constraints = nothing`: constraints specification object, optional, see `@constraints`
- `meta  = nothing`: meta specification object, optional, may be required for some models, see `@meta`
- `options = nothing`: model creation options, optional, see `ModelInferenceOptions`
- `returnvars = nothing`: return structure info, optional, by default creates observables for all random variables that return posteriors at last vmp iteration, see below for more information
- `historyvars = nothing`: history structure info, optional, defaults to no history, see below for more information
- `keephistory = nothing`: history buffer size, defaults to empty buffer, see below for more information
- `iterations = nothing`: number of iterations, optional, defaults to `nothing`, the inference engine does not distinguish between variational message passing or Loopy belief propagation or expectation propagation iterations, see below for more information
- `free_energy = false`: compute the Bethe free energy, optional, defaults to false. Can be passed a floating point type, e.g. `Float64`, for better efficiency, but disables automatic differentiation packages, such as ForwardDiff.jl
- `free_energy_diagnostics = BetheFreeEnergyDefaultChecks`: free energy diagnostic checks, optional, by default checks for possible `NaN`s and `Inf`s. `nothing` disables all checks.
- `autostart = true`: specifies whether to call `RxInfer.start` on the created engine automatically or not
- `showprogress = false`: show progress module, optional, defaults to false
- `events = nothing`: inference cycle events, optional, see below for more info
- `callbacks = nothing`: inference cycle callbacks, optional, see below for more info
- `uselock = false`: specifies either to use the lock structure for the inference or not, if set to true uses `Base.Threads.SpinLock`. Accepts custom `AbstractLock`.
- `warn = true`: enables/disables warnings

## Note on NamedTuples

When passing `NamedTuple` as a value for some argument, make sure you use a trailing comma for `NamedTuple`s with a single entry. The reason is that Julia treats `historyvars = (x = KeepLast())` and 
`historyvars = (x = KeepLast(), )` expressions differently. First expression creates (or **overwrites!**) new local/global variable named `x` with contents `KeepLast()`. The second expression (note traling comma)
creates `NamedTuple` with `x` as a key and `KeepLast()` as a value assigned for this key.

## Extended information about some of the arguments

- ### `model`

The `model` argument accepts a `ModelGenerator` as its input. The easiest way to create the `ModelGenerator` is to use the `@model` macro. 
For example:

```julia
@model function coin_toss(some_argument, some_keyword_argument = 3)
   ...
end

result = rxinference(
    model = coin_toss(some_argument; some_keyword_argument = 3)
)
```

**Note**: The `model` keyword argument does not accept a `FactorGraphModel` instance as a value, as it needs to inject `constraints` and `meta` during the inference procedure.

- ### `initmarginals`

In general for variational message passing every marginal distribution in a model needs to be pre-initialised. In practice, however, for many models it is sufficient enough to initialise only a small subset of variables in the model.

- ### `initmessages`

Loopy belief propagation may need some messages in a model to be pre-initialised.

- ### `autoupdates`

See `@autoupdates` for more information.

- ### `options`

- `limit_stack_depth`: limits the stack depth for computing messages, helps with `StackOverflowError` for some huge models, but reduces the performance of inference backend. Accepts integer as an argument that specifies the maximum number of recursive depth. Lower is better for stack overflow error, but worse for performance.
- `pipeline`: changes the default pipeline for each factor node in the graph
- `global_reactive_scheduler`: changes the scheduler of reactive streams, see Rocket.jl for more info, defaults to no scheduler

- ### `returnvars`

`returnvars` accepts a tuple of symbols and specifies the latent variables of interests. For each symbol in the `returnvars` specification the `rxinference` function will prepare an observable 
stream (see `Rocket.jl`) of posterior updates. An agent may subscribe on the new posteriors events and perform some actions.
For example:

```julia
engine = rxinference(
    ...,
    returnvars = (:x, :τ),
    autostart  = false
)

x_subscription = subscribe!(engine.posteriors[:x], (update) -> println("x variable has been updated: ", update))
τ_subscription = subscribe!(engine.posteriors[:τ], (update) -> println("τ variable has been updated: ", update))

RxInfer.start(engine)

...

unsubscribe!(x_subscription)
unsubscribe!(τ_subscription)

RxInfer.stop(engine)
```

- ### `historyvars`

`historyvars` specifies the variables of interests and the amount of information to keep in history about the posterior updates. The specification is similar to the `returnvars` in the `inference` procedure.
The `historyvars` requires `keephistory` to be greater than zero.

`historyvars` accepts a `NamedTuple` or `Dict` or return var specification. There are two specifications:
- `KeepLast`: saves the last update for a variable, ignoring any intermediate results during iterations
- `KeepEach`: saves all updates for a variable for all iterations

Example: 

```julia
result = rxinference(
    ...,
    historyvars = (
        x = KeepLast(),
        τ = KeepEach()
    ),
    keephistory = 10
)
```

It is also possible to set iether `historyvars = KeepLast()` or `historyvars = KeepEach()` that acts as an alias and sets the given option for __all__ random variables in the model.

# Example: 

```julia
result = rxinference(
    ...,
    historyvars = KeepLast(),
    keephistory = 10
)
```

- ### `keep_history`

Specifies the buffer size for the updates history both for the `historyvars` and the `free_energy` buffers.. 

- ### `iterations`

Specifies the number of variational (or loopy BP) iterations. By default set to `nothing`, which is equivalent of doing 1 iteration. 

- ### `free_energy` 

This setting specifies whenever the `inference` function should create an observable of Bethe Free Energy (BFE) values. The BFE observable returns a new computed value for each VMP iteration.
Note, however, that it may be not possible to compute BFE values for every model. If `free_energy = true` and `keephistory > 0` the engine exposes extra fields to access the history of the Bethe free energy updates:

- `engine.free_energy_history`: Returns a free energy history averaged over the VMP iterations
- `engine.free_energy_final_only_history`: Returns a free energy history of values computed on last VMP iterations for every observation
- `engine.free_energy_raw_history`: Returns a raw free energy history

Additionally, the argument may accept a floating point type, instead of a `Bool` value. Using this option, e.g.`Float64`, improves performance of Bethe Free Energy computation, but restricts using automatic differentiation packages.

- ### `free_energy_diagnostics`

This settings specifies either a single or a tuple of diagnostic checks for Bethe Free Energy values stream. By default checks for `NaN`s and `Inf`s. See also [`BetheFreeEnergyCheckNaNs`](@ref) and [`BetheFreeEnergyCheckInfs`](@ref).
Pass `nothing` to disable any checks.

- ### `events`

The engine from the `rxinference` function has its own lifecycle. The events can be listened by subscribing to the `engine.events` field. E.g.

```julia
engine = rxinference(
    ...,
    autostart = false
)

subscription = subscribe!(engine.events, (event) -> println(event))

RxInfer.start(engine)
```

By default all events are disabled, in order to enable an event its identifier must be listed in the `Val` tuple of symbols passed to the `events` keyword arguments.

```julia
engine = rxinference(
    events = Val((:on_new_data, :before_history_save, :after_history_save))
)
```

The list of all possible events and their event data is present below (see `RxInferenceEvent` for more information about the type of event data):

- `on_new_data`:           args: (model::FactorGraphModel, data)
- `before_iteration`       args: (model::FactorGraphModel, iteration)
- `before_auto_update`     args: (model::FactorGraphModel, iteration, auto_updates)
- `after_auto_update`      args: (model::FactorGraphModel, iteration, auto_updates)
- `before_data_update`     args: (model::FactorGraphModel, iteration, data)
- `after_data_update`      args: (model::FactorGraphModel, iteration, data)
- `after_iteration`        args: (model::FactorGraphModel, iteration)
- `before_history_save`    args: (model::FactorGraphModel, )
- `after_history_save`     args: (model::FactorGraphModel, )
- `on_tick`                args: (model::FactorGraphModel, )
- `on_error`               args: (model::FactorGraphModel, err)
- `on_complete`            args: (model::FactorGraphModel, )

- ### `callbacks`

The `rxinference` function has its own lifecycle. The user is free to provide some (or none) of the callbacks to inject some extra logging or other procedures in the preparation of the inference engine.
To inject extra procedures during the inference use the `events`. Here is the example of the `callbacks`

```julia
result = rxinference(
    ...,
    callbacks = (
        after_model_creation = (model, returnval) -> println("The model has been created. Number of nodes: \$(length(getnodes(model)))"),
    )
)
```

The `callbacks` keyword argument accepts a named-tuple of 'name = callback' pairs. 
The list of all possible callbacks and their input arguments is present below:

- `before_model_creation`:    args: ()
- `after_model_creation`:     args: (model::FactorGraphModel, returnval)
- `before_autostart`:         args: (engine::RxInferenceEngine)
- `after_autostart`:          args: (engine::RxInferenceEngine)
"""
function rxinference(;
    model::ModelGenerator,
    data = nothing,
    datastream = nothing,
    initmarginals = nothing,
    initmessages = nothing,
    autoupdates = nothing,
    constraints = nothing,
    meta = nothing,
    options = nothing,
    returnvars = nothing,
    historyvars = nothing,
    keephistory = nothing,
    iterations = nothing,
    free_energy = false,
    free_energy_diagnostics = BetheFreeEnergyDefaultChecks,
    autostart = true,
    events = nothing,
    callbacks = nothing,
    uselock = false,
    warn = true
)
    __inference_check_dicttype(:callbacks, callbacks)

    # Check for available callbacks
    if warn && !isnothing(callbacks)
        for key in keys(callbacks)
            if key ∉ (:before_model_creation, :after_model_creation, :before_autostart, :after_autostart)
                @warn "Unknown callback specification: $(key). Available callbacks: before_model_creation, after_model_creation, before_autostart, after_autostart. Set `warn = false` to supress this warning."
            end
        end
    end

    # The `rxinference` support both static `data` and dynamic `datastream`
    if !isnothing(data) && !isnothing(datastream) # Ensure that only one of them set
        error("`data` and `datastream` keyword arguments cannot be used together")
    elseif isnothing(data) && isnothing(datastream) # Ensure that at least one of them set
        error("The `rxinference` function requires either `data` or `datastream` keyword argument to be non-empty.")
    end

    # In case if `data` is used we cast to a synchronous `datastream` with zip operator
    _datastream, _T = if isnothing(datastream) && !isnothing(data)
        __inference_check_dicttype(:data, data)

        names  = tuple(keys(data)...)
        items  = tuple(values(data)...)
        stream = labeled(Val(names), iterable(zip(items...)))
        etype  = NamedTuple{names, Tuple{eltype.(items)...}}

        stream, etype
    else
        eltype(datastream) <: NamedTuple || error("`eltype` of the `datastream` must be a `NamedTuple`")
        datastream, eltype(datastream)
    end

    datavarnames = fields(_T)::NTuple
    N            = length(datavarnames) # should be static

    _options = convert(ModelInferenceOptions, options)

    inference_invoke_callback(callbacks, :before_model_creation)
    _model, _returnval = create_model(model, constraints = constraints, meta = meta, options = _options)
    inference_invoke_callback(callbacks, :after_model_creation, _model, _returnval)
    vardict = getvardict(_model)

    # At the very beginning we try to preallocate handles for the `datavar` labels that are present in the `T` (from `datastream`)
    # This is not very type-styble-friendly but we do it once and it should pay-off in the inference procedure
    datavars = ntuple(N) do i
        datavarname = datavarnames[i]
        hasdatavar(_model, datavarname) || error("The `datastream` produces data for `$(datavarname)`, but the model does not have a datavar named `$(datavarname)`")
        return _model[datavarname]::DataVariable
    end

    # Second we check autoupdates and pregenerate all necessary structures here
    _autoupdates = map((autoupdate) -> autoupdate(_model), something(autoupdates, ()))

    __inference_check_dicttype(:initmarginals, initmarginals)
    __inference_check_dicttype(:initmessages, initmessages)

    # If everything is ok with `datavars` and `redirectvars` next step is to initialise marginals and messages in the model
    # This happens only once at the creation, we do not reinitialise anything if the inference has been stopped and resumed with the `stop` and `start` functions
    if !isnothing(initmarginals)
        for (variable, initvalue) in pairs(initmarginals)
            if haskey(vardict, variable)
                assign_marginal!(vardict[variable], initvalue)
            elseif warn
                @warn "`initmarginals` object has `$(variable)` specification, but model has no variable named `$(variable)`. Use `warn = false` to suppress this warning."
            end
        end
    end

    if !isnothing(initmessages)
        for (variable, initvalue) in pairs(initmessages)
            if haskey(vardict, variable)
                assign_message!(vardict[variable], initvalue)
            elseif warn
                @warn "`initmessages` object has `$(variable)` specification, but model has no variable named `$(variable)`. Use `warn = false` to suppress this warning."
            end
        end
    end

    # `iterations` might be set to `nothing` in which case we assume `1` iteration
    _iterations = something(iterations, 1)
    _iterations isa Integer || error("`iterations` argument must be of type Integer or `nothing`")
    _iterations > 0 || error("`iterations` arguments must be greater than zero")

    _keephistory = something(keephistory, 0)
    _keephistory isa Integer || error("`keephistory` argument must be of type Integer or `nothing`")
    _keephistory >= 0 || error("`keephistory` arguments must be greater than or equal to zero")

    # `tickscheduler` defines a moment when we send new posteriors in the `posteriors` streams
    tickscheduler = PendingScheduler()

    # Here we prepare our free energy streams (if requested)
    fe_actor     = nothing
    fe_objective = nothing
    fe_scheduler = nothing
    fe_source    = nothing

    # `free_energy` may accept a type specification (e.g. `Float64`) in which case it counts as `true` as well
    # An explicit type specification makes `fe_source` be a bit more efficient, but makes it hard to differentiate the model
    is_free_energy, S, FE_T = unwrap_free_energy_option(free_energy)

    if is_free_energy
        if _keephistory > 0
            fe_actor = ScoreActor(S, _iterations, _keephistory)
        end
        fe_scheduler = PendingScheduler()
        fe_objective = BetheFreeEnergy(BetheFreeEnergyDefaultMarginalSkipStrategy, fe_scheduler, free_energy_diagnostics)
        fe_source    = score(_model, FE_T, fe_objective)
    end

    # Use `__check_has_randomvar` to filter out unknown or non-random variables in the `returnvars` and `historyvars` specification
    __check_has_randomvar(object, vardict, key) = begin
        haskey_check   = haskey(vardict, key)
        israndom_check = haskey_check ? israndom(vardict[key]) : false
        if warn && !haskey_check
            @warn "`$(object)` object has `$(key)` specification, but model has no variable named `$(key)`. The `$(key)` specification is ignored. Use `warn = false` to suppress this warning."
        elseif warn && haskey_check && !israndom_check
            @warn "`$(object)` object has `$(key)` specification, but model has no **random** variable named `$(key)`. The `$(key)` specification is ignored. Use `warn = false` to suppress this warning."
        end
        return haskey_check && israndom_check
    end

    # We check if `returnvars` argument is empty, in which case we return names of all random (non-proxy) variables in the model
    if isnothing(returnvars)
        returnvars = [name(variable) for (variable, value) in pairs(vardict) if (israndom(value) && !isanonymous(value))]
    end

    eltype(returnvars) === Symbol || error("`returnvars` must contain a list of symbols") # TODO?

    returnvars = filter((varkey) -> __check_has_randomvar(:returnvars, vardict, varkey), returnvars)

    __inference_check_itertype(:returnvars, returnvars)

    # `rxinference` by default does not keep track of marginals updates history
    # If user specifies `keephistory` keyword argument
    if _keephistory > 0
        if isnothing(historyvars)
            # First what we do - we check if `historyvars` is nothing 
            # In which case we mirror the `returnvars` specication and use either `KeepLast()` or `KeepEach` (depending on the iterations spec)
            historyoption = _iterations > 1 ? KeepEach() : KeepLast()
            historyvars   = Dict(name => historyoption for name in returnvars)
        elseif historyvars === KeepEach() || historyvars === KeepLast()
            # Second we check if it is one of the two possible global values: `KeepEach` and `KeepLast`. 
            # If so, we replace it with either `KeepEach` or `KeepLast` for each random and not-proxied variable in a model
            historyvars = Dict(name(variable) => historyvars for (variable, value) in pairs(vardict) if (israndom(value) && !isproxy(value)))
        end

        historyvars = Dict((varkey => value) for (varkey, value) in pairs(historyvars) if __check_has_randomvar(:historyvars, vardict, varkey))

        __inference_check_dicttype(:historyvars, historyvars)
    else
        if !isnothing(historyvars) && warn
            @warn "`historyvars` keyword argument requires `keephistory > 0`. Ignoring `historyvars`. Use `warn = false` to suppress this warning."
            historyvars = nothing
        end
    end

    # Here we finally create structures for updates history 
    historyactors = nothing
    history       = nothing

    if !isnothing(historyvars) && _keephistory > 0
        historyactors = Dict(name => make_actor(vardict[name], historyoption, _iterations) for (name, historyoption) in pairs(historyvars))
        history       = Dict(name => CircularBuffer(_keephistory) for (name, _) in pairs(historyvars))
    end

    # At this point we must have properly defined and fixed `returnvars` and `historyvars` objects

    # For each random variable entry in `returnvars` specification we create a boolean flag to track their updates
    updateflags = Dict(variable => MarginalHasBeenUpdated(false) for variable in returnvars)

    # `posteriors` returns a `stream` for each entry in the `returnvars`
    posteriors = Dict(variable => obtain_marginal(vardict[variable]) |> schedule_on(tickscheduler) |> map(Any, getdata) for variable in returnvars)

    _events        = Subject(RxInferenceEvent)
    _enabledevents = something(events, Val(()))

    if !(_enabledevents isa Val) || !(unval(_enabledevents) isa Tuple)
        error("`events` keyword argument must be a `Val` of tuple of symbols")
    elseif length(unval(_enabledevents)) > 0 && !(eltype(unval(_enabledevents)) === Symbol)
        error("`events` keyword argument must be a `Val` of tuple of symbols")
    end

    # By default we do not use any lock synchronization
    _ticklock = nothing

    # Check the lock
    if uselock === true
        _ticklock = Base.Threads.SpinLock()
    elseif uselock !== false # This check makes sense because `uselock` is not necessarily of the `Bool` type
        _ticklock = uselock
    end

    engine = RxInferenceEngine(
        _T,
        _datastream,
        tickscheduler,
        datavars,
        posteriors,
        updateflags,
        history,
        historyactors,
        _autoupdates,
        fe_actor,
        fe_scheduler,
        fe_objective,
        fe_source,
        _iterations,
        _model,
        _returnval,
        _enabledevents,
        _events,
        _ticklock
    )

    if autostart
        inference_invoke_callback(callbacks, :before_autostart, engine)
        start(engine)
        inference_invoke_callback(callbacks, :after_autostart, engine)
    end

    return engine
end
