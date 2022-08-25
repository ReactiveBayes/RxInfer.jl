export inference, InferenceResult, KeepEach, KeepLast

import ProgressMeter

obtain_marginal(variable::AbstractVariable)                   = getmarginal(variable)
obtain_marginal(variables::AbstractArray{<:AbstractVariable}) = getmarginals(variables)

assign_marginal!(variables::AbstractArray{<:AbstractVariable}, marginals) = setmarginals!(variables, marginals)
assign_marginal!(variable::AbstractVariable, marginal)                    = setmarginal!(variable, marginal)

assign_message!(variables::AbstractArray{<:AbstractVariable}, messages) = setmessages!(variables, messages)
assign_message!(variable::AbstractVariable, message)                    = setmessage!(variable, message)

struct KeepEach end
struct KeepLast end

make_actor(::RandomVariable, ::KeepEach)                       = keep(Marginal)
make_actor(::Array{<:RandomVariable, N}, ::KeepEach) where {N} = keep(Array{Marginal, N})
make_actor(x::AbstractArray{<:RandomVariable}, ::KeepEach)     = keep(typeof(similar(x, Marginal)))

make_actor(::RandomVariable, ::KeepLast)                   = storage(Marginal)
make_actor(x::AbstractArray{<:RandomVariable}, ::KeepLast) = buffer(Marginal, size(x))

## Inference ensure update

mutable struct MarginalHasBeenUpdated
    updated::Bool
end

__unset_updated!(updated::MarginalHasBeenUpdated) = updated.updated = false
__set_updated!(updated::MarginalHasBeenUpdated)   = updated.updated = true

# This creates a `tap` operator that will set the `updated` flag to true. 
# Later on we check flags and `unset!` them after the `update!` procedure
ensure_update(model::FactorGraphModel, callback, variable_name::Symbol, updated::MarginalHasBeenUpdated)  = tap((update) -> begin
    __set_updated!(updated)
    callback(model, variable_name, update)
end)
ensure_update(model::FactorGraphModel, ::Nothing, variable_name::Symbol, updated::MarginalHasBeenUpdated) = tap((_) -> __set_updated!(updated)) # If `callback` is nothing we simply set updated flag

## Extra error handling

__inference_process_error(error) = rethrow(error)

function __inference_process_error(err::StackOverflowError)
    @error """
    Stack overflow error occurred during the inference procedure. 
    The inference engine may execute message update rules recursively, hence, the model graph size might be causing this error. 
    To resolve this issue, try using `limit_stack_depth` option for model creation. See `?model_options` and `?inference` documentation for more details.
    The `limit_stack_depth` option does not help against over stack overflow errors that might hapenning outside of the model creation or message update rules execution.
    """
    rethrow(err) # Shows the original stack trace
end

__inference_check_dicttype(::Symbol, ::Union{Nothing, NamedTuple, Dict}) = nothing

function __inference_check_dicttype(keyword::Symbol, input::T) where {T}
    error(
        """
        Keyword argument `$(keyword)` expects either `Dict` or `NamedTuple` as an input, but a value of type `$(T)` has been used.
        If you specify a `NamedTuple` with a single entry - make sure you put a trailing comma at then end, e.g. `(x = something, )`. 
        Note: Julia's parser interprets `(x = something)` and (x = something, ) differently. 
            The first expression defines (or **overwrites!**) the local/global variable named `x` with `something` as a content. 
            The second expression defines `NamedTuple` with `x` as a key and `something` as a value.
        """
    )
end
##

"""
    InferenceResult

This structure is used as a return value from the [`inference`](@ref) function. 

# Fields

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
    if !isnothing(getfield(result, :free_energy))
        print(io, "-----------------------------------------\n")
        print(io, "Free Energy: ")
        print(IOContext(io, :compact => true, :limit => true, :displaysize => (1, 80)), result.free_energy)
        print(io, "\n")
    end
    maxdisplay = 80
    maxlen     = maximum(p -> length(string(first(p))), pairs(result.posteriors), init = 0)
    print(io, "-----------------------------------------\n")
    for (key, value) in pairs(result.posteriors)
        print(io, "$(rpad(key, maxlen)) = ")
        strval    = repr(value, context = :limit => true)
        lastindex = last(collect(Iterators.take(eachindex(strval), maxdisplay)))
        print(io, view(strval, 1:lastindex))
        if lastindex >= maxdisplay
            print(io, "...")
        end
        print(io, "\n")
    end
end

function Base.getproperty(result::InferenceResult, property::Symbol)
    if property === :free_energy && getfield(result, :free_energy) === nothing
        error(
            """
            Bethe Free Energy has not been computed. 
            Use `free_energy = true` keyword argument for the `inference` function to compute Bethe Free Energy values.
            """
        )
    else
        return getfield(result, property)
    end
    return getfield(result, property)
end

__inference_invoke_callback(callback, args...)  = callback(args...)
__inference_invoke_callback(::Nothing, args...) = begin end

inference_invoke_callback(callbacks, name, args...) =
    __inference_invoke_callback(inference_get_callback(callbacks, name), args...)
inference_invoke_callback(::Nothing, name, args...) = begin end

inference_get_callback(callbacks, name) = get(() -> nothing, callbacks, name)
inference_get_callback(::Nothing, name) = nothing

unwrap_free_energy_option(option::Bool)                      = (option, Real, InfCountingReal)
unwrap_free_energy_option(option::Type{T}) where {T <: Real} = (true, T, InfCountingReal{T})

"""
    inference(
        model::ModelGenerator; 
        data,
        initmarginals           = nothing,
        initmessages            = nothing,
        constraints             = nothing,
        meta                    = nothing,
        options                 = (;),
        returnvars              = nothing, 
        iterations              = nothing,
        free_energy             = false,
        free_energy_diagnostics = BetheFreeEnergyDefaultChecks,
        showprogress            = false,
        callbacks               = nothing,
    )

This function provides a generic (but somewhat limited) way to perform probabilistic inference in ReactiveMP.jl. Returns `InferenceResult`.

## Arguments

For more information about some of the arguments, please check below.

- `model::ModelGenerator`: specifies a model generator, with the help of the `Model` function
- `data`: `NamedTuple` or `Dict` with data, required
- `initmarginals = nothing`: `NamedTuple` or `Dict` with initial marginals, optional, defaults to nothing
- `initmessages = nothing`: `NamedTuple` or `Dict` with initial messages, optional, defaults to nothing
- `constraints = nothing`: constraints specification object, optional, see `@constraints`
- `meta  = nothing`: meta specification object, optional, may be required for some models, see `@meta`
- `options = (;)`: model creation options, optional, see `model_options`
- `returnvars = nothing`: return structure info, optional, defaults to return everything at each iteration, see below for more information
- `iterations = nothing`: number of iterations, optional, defaults to `nothing`, we do not distinguish between variational message passing or Loopy belief propagation or expectation propagation iterations, see below for more information
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

The `model` argument accepts a `ModelGenerator` as its input. The easiest way to create the `ModelGenerator` is to use the `Model` function. The `Model` function accepts a model name as its first argument and the rest is passed directly to the model constructor.
For example:

```julia
result = inference(
    # Creates `coin_toss(some_argument, some_keyword_argument = 3)`
    model = Model(coin_toss, some_argument; some_keyword_argument = 3)
)
```

**Note**: The `model` keyword argument does not accept a `FactorGraphModel` instance as a value, as it needs to inject `constraints` and `meta` during the inference procedure.

- ### `initmarginals`

In general for variational message passing every marginal distribution in a model needs to be pre-initialised. In practice, however, for many models it is sufficient enough to initialise only a small subset of variables in the model.

- ### `initmessages`

Loopy belief propagation may need some messages in a model to be pre-initialised.

- ### `options`

See `?model_options`.

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

The list of all possible callbacks is present below:

- `:on_marginal_update`:    args: (model::FactorGraphModel, name::Symbol, update)
- `:before_model_creation`: args: ()
- `:after_model_creation`:  args: (model::FactorGraphModel)
- `:before_inference`:      args: (model::FactorGraphModel)
- `:before_iteration`:      args: (model::FactorGraphModel, iteration::Int)
- `:before_data_update`:    args: (model::FactorGraphModel, data)
- `:after_data_update`:     args: (model::FactorGraphModel, data)
- `:after_iteration`:       args: (model::FactorGraphModel, iteration::Int)
- `:after_inference`:       args: (model::FactorGraphModel)

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
    options = (;),
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

    inference_invoke_callback(callbacks, :before_model_creation)
    fmodel, freturval = create_model(model, constraints, meta, options)
    inference_invoke_callback(callbacks, :after_model_creation, fmodel)
    vardict = ReactiveMP.getvardict(fmodel)

    # First what we do - we check if `returnvars` is nothing or one of the two possible values: `KeepEach` and `KeepLast`. 
    # If so, we replace it with either `KeepEach` or `KeepLast` for each random and not-proxied variable in a model
    if returnvars === nothing || returnvars === KeepEach() || returnvars === KeepLast()
        # Checks if the first argument is `nothing`, in which case returns the second argument
        returnoption = something(returnvars, iterations isa Number ? KeepEach() : KeepLast())
        returnvars =
            Dict(
                variable => returnoption for (variable, value) in pairs(vardict) if (israndom(value) && !isproxy(value))
            )
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
    actors = Dict(
        variable => make_actor(vardict[variable], value) for
        (variable, value) in pairs(returnvars) if __check_has_randomvar(vardict, variable)
    )

    # At third, for each random variable entry we create a boolean flag to track their updates
    updates = Dict(variable => MarginalHasBeenUpdated(false) for (variable, _) in pairs(actors))

    try
        on_marginal_update = inference_get_callback(callbacks, :on_marginal_update)
        subscriptions      = Dict(variable => subscribe!(obtain_marginal(vardict[variable]) |> ensure_update(fmodel, on_marginal_update, variable, updates[variable]), actor) for (variable, actor) in pairs(actors))

        fe_actor        = nothing
        fe_subscription = VoidTeardown()

        is_free_energy, S, T = unwrap_free_energy_option(free_energy)

        if is_free_energy
            fe_actor        = ScoreActor(S)
            fe_subscription = subscribe!(score(T, BetheFreeEnergy(BetheFreeEnergyDefaultMarginalSkipStrategy, free_energy_diagnostics), fmodel), fe_actor)
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
                haskey(data, varname) || error(
                    "Data entry `$(varname)` is missing in `data` argument. Double check `data = ($(varname) = ???, )`"
                )
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

        _iterations = something(iterations, 1)
        _iterations isa Integer || error("`iterations` argument must be of type Integer or `nothing`")
        _iterations > 0 || error("`iterations` arguments must be greater than zero")

        p = showprogress ? ProgressMeter.Progress(_iterations) : nothing

        for iteration in 1:_iterations
            inference_invoke_callback(callbacks, :before_iteration, fmodel, iteration)
            inference_invoke_callback(callbacks, :before_data_update, fmodel, data)
            for (key, value) in fdata
                update!(vardict[key], value)
            end
            inference_invoke_callback(callbacks, :after_data_update, fmodel, data)
            not_updated = filter((pair) -> !last(pair).updated, updates)
            if length(not_updated) !== 0
                names = join(keys(not_updated), ", ")
                error(
                    """
                  Variables [ $(names) ] have not been updated after a single inference iteration. 
                  Therefore, make sure to initialize all required marginals and messages. See `initmarginals` and `initmessages` keyword arguments for the `inference` function. 
              """
                )
            end
            for (_, update_flag) in pairs(updates)
                __unset_updated!(update_flag)
            end
            if !isnothing(p)
                ProgressMeter.next!(p)
            end
            inference_invoke_callback(callbacks, :after_iteration, fmodel, iteration)
        end

        for (_, subscription) in pairs(subscriptions)
            unsubscribe!(subscription)
        end

        unsubscribe!(fe_subscription)

        posterior_values = Dict(variable => getdata(getvalues(actor)) for (variable, actor) in pairs(actors))
        fe_values        = fe_actor !== nothing ? getvalues(fe_actor) : nothing

        inference_invoke_callback(callbacks, :after_inference, fmodel)

        return InferenceResult(posterior_values, fe_values, fmodel, freturval)
    catch error
        __inference_process_error(error)
    end
end
