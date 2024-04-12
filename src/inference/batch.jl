"""
    InferenceResult

This structure is used as a return value from the [`infer`](@ref) function for static datasets. 

# Public Fields

- `posteriors`: `Dict` or `NamedTuple` of 'random variable' - 'posterior' pairs. See the `returnvars` argument for [`infer`](@ref).
- `predictions`: (optional) `Dict` or `NamedTuple` of 'data variable' - 'prediction' pairs. See the `predictvars` argument for [`infer`](@ref).
- `free_energy`: (optional) An array of Bethe Free Energy values per VMP iteration. See the `free_energy` argument for [`infer`](@ref).
- `model`: `FactorGraphModel` object reference.
- `error`: (optional) A reference to an exception, that might have occurred during the inference. See the `catch_exception` argument for [`infer`](@ref).

See also: [`infer`](@ref)
"""
struct InferenceResult{P, A, F, M, E}
    posteriors  :: P
    predictions :: A
    free_energy :: F
    model       :: M
    error       :: E
end

Base.iterate(results::InferenceResult)      = iterate((getfield(results, :posteriors), getfield(results, :predictions), getfield(results, :free_energy), getfield(results, :model), getfield(results, :returnval), getfield(results, :error)))
Base.iterate(results::InferenceResult, any) = iterate((getfield(results, :posteriors), getfield(results, :predictions), getfield(results, :free_energy), getfield(results, :model), getfield(results, :returnval), getfield(results, :error)), any)

"""
Checks if the `InferenceResult` object does not contain an error. 
"""
issuccess(result::InferenceResult) = !iserror(result)

"""
Checks if the `InferenceResult` object contains an error. 
"""
iserror(result::InferenceResult) = !isnothing(result.error)

function Base.show(io::IO, result::InferenceResult)
    print(io, "Inference results:\n")

    lcolumnlen = 18 # Defines the padding for the "left" column of the output

    print(io, rpad("  Posteriors", lcolumnlen), " | ")
    print(io, "available for (")
    join(io, keys(getfield(result, :posteriors)), ", ")
    print(io, ")\n")

    if !isempty(getfield(result, :predictions))
        print(io, rpad("  Predictions", lcolumnlen), " | ")
        print(io, "available for (")
        join(io, keys(getfield(result, :predictions)), ", ")
        print(io, ")\n")
    end

    if !isnothing(getfield(result, :free_energy))
        print(io, rpad("  Free Energy:", lcolumnlen), " | ")
        print(IOContext(io, :compact => true, :limit => true, :displaysize => (1, 80)), result.free_energy)
        print(io, "\n")
    end

    if iserror(result)
        print(
            io,
            "[ WARN ] An error has occurred during the inference procedure. The result might not be complete. You can use the `.error` field to access the error and its backtrace. Use `Base.showerror` function to display the error."
        )
    end
end

function Base.showerror(result::InferenceResult)
    return Base.showerror(stderr, result)
end

function Base.showerror(io::IO, result::InferenceResult)
    if iserror(result)
        error, backtrace = result.error
        println(io, error, "\n")
        show(io, "text/plain", stacktrace(backtrace))
    else
        print(io, "The inference has completed successfully.")
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

function batch_inference(;
    # `model` must be a materialized graph object from GraphPPL 
    model::ModelGenerator,
    # NamedTuple or Dict with data, optional if predictvars are specified
    data = nothing,
    # Initialization specification object
    initialization = nothing,
    # Constraints specification object
    constraints = nothing,
    # Meta specification object
    meta = nothing,
    # Model creation options
    options = nothing,
    # Return structure info, optional, defaults to return everything at each iteration
    returnvars = nothing,
    # Prediction structure info, optional, defaults to return everything at each iteration
    predictvars = nothing,
    # Number of iterations, defaults to 1, we do not distinguish between VMP or Loopy belief or EP iterations
    iterations = nothing,
    # Do we compute FE, optional, defaults to false 
    # Can be passed a floating point type, e.g. `Float64`, for better efficiency, but disables automatic differentiation packages, such as ForwardDiff.jl
    free_energy = false,
    # Default BFE stream checks
    free_energy_diagnostics = DefaultObjectiveDiagnosticChecks,
    # Show progress module, optional, defaults to false
    showprogress = false,
    # Inference cycle callbacks
    callbacks = nothing,
    # Addons specification
    addons = nothing,
    # Inference postprocessing option
    postprocess = DefaultPostprocess(),
    # warn, optional, defaults to true
    warn = true,
    # catch exceptions during the inference procedure, optional, defaults to false
    catch_exception = false
)
    _options = convert(ReactiveMPInferenceOptions, options)
    # If the `options` does not have `warn` key inside, override it with the keyword `warn`
    if isnothing(options) || !haskey(options, :warn)
        _options = setwarn(_options, warn)
    end

    # Override `options` addons if the `addons` keyword argument is present 
    if !isnothing(addons)
        if warn && !isnothing(getaddons(_options))
            @warn "Both `addons = ...` and `options = (addons = ..., )` specify a value for the `addons`. Ignoring the `options` setting. Set `warn = false` to supress this warning."
        end
        _options = setaddons(_options, addons)
    end

    # We create a model with the `GraphPPL` package and insert a certain RxInfer related 
    # plugins which include the VI plugin, meta plugin and the ReactiveMP integration plugin
    modelplugins = GraphPPL.PluginsCollection(
        GraphPPL.VariationalConstraintsPlugin(constraints), GraphPPL.MetaPlugin(meta), RxInfer.InitializationPlugin(initialization), RxInfer.ReactiveMPInferencePlugin(_options)
    )

    is_free_energy, S = unwrap_free_energy_option(free_energy)

    if is_free_energy
        fe_objective = BetheFreeEnergy(S)
        modelplugins = modelplugins + ReactiveMPFreeEnergyPlugin(fe_objective)
    end

    # The `_model` here still must be a `ModelGenerator`
    _model = GraphPPL.with_plugins(model, modelplugins)

    infer_check_dicttype(:data, data)

    # If `predictvars` is specified implicitly as `KeepEach` or `KeepLast`, we replace it with the same value for each data variable
    if (predictvars === KeepEach() || predictvars === KeepLast())
        if !isnothing(data)
            predictoption = predictvars
            predictvars = Dict(variable => predictoption for (variable, value) in pairs(data))
        else # else we throw an error
            error("`predictvar` is specified as `$(predictvars)`, but `data` is not provided. Make sure to provide `data` or specify `predictvars` explicitly.")
        end
        # If `predictvar` is specified, but `data` is not, we initialize the `data` with missing values
    elseif !isnothing(predictvars) && isnothing(data)
        data = Dict(variable => missing for (variable, value) in pairs(predictvars))
        # If `predictvar` is not specified, but `data` is, we initialize the `predictvars` with `KeepLast` or `KeepEach` depending on the `iterations` value
        # But only if the data has missing values in it
    elseif isnothing(predictvars) && !isnothing(data)
        predictoption = iterations isa Number ? KeepEach() : KeepLast()
        predictvars = Dict(variable => predictoption for (variable, value) in pairs(data) if inference_check_dataismissing(value))
        # If both `predictvar` and `data` are specified we double check if there are some entries in the `predictvars`
        # which are not specified in the `data` and inject them
        # We do the same the other way around for the `data` entries which are not specified in the `predictvars`
    elseif !isnothing(predictvars) && !isnothing(data)
        for (variable, _) in pairs(predictvars)
            if !haskey(data, variable)
                data = merge(data, Dict(variable => missing))
            end
        end
        for (variable, value) in pairs(data)
            if !haskey(predictvars, variable) && inference_check_dataismissing(value)
                predictoption = iterations isa Number ? KeepEach() : KeepLast()
                predictvars = merge(predictvars, Dict(variable => predictoption))
            end
        end
    end

    infer_check_dicttype(:predictvars, predictvars)

    inference_invoke_callback(callbacks, :before_model_creation)
    fmodel = create_model(_model | data)
    inference_invoke_callback(callbacks, :after_model_creation, fmodel)
    vardict = getvardict(fmodel)
    vardict = GraphPPL.variables(vardict) # TODO bvdmitri, should work recursively as well

    # First what we do - we check if `returnvars` is nothing or one of the two possible values: `KeepEach` and `KeepLast`. 
    # If so, we replace it with either `KeepEach` or `KeepLast` for each random and not-proxied variable in a model
    if isnothing(returnvars) || returnvars === KeepEach() || returnvars === KeepLast()
        # Checks if the first argument is `nothing`, in which case returns the second argument
        returnoption = something(returnvars, iterations isa Number ? KeepEach() : KeepLast())
        returnvars   = Dict(variable => returnoption for (variable, value) in pairs(vardict) if (israndom(value) && !isanonymous(value)))
    end

    infer_check_dicttype(:returnvars, returnvars)

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

    # Use `__check_has_prediction` to filter out unknown predictions variables in the `predictvar` specification
    __check_has_prediction(vardict, variable) = begin
        haskey_check = haskey(vardict, variable)
        isdata_check = haskey_check ? isdata(vardict[variable]) : false
        if warn && !haskey_check
            @warn "`predictvars` object has `$(variable)` specification, but model has no variable named `$(variable)`. The `$(variable)` specification is ignored. Use `warn = false` to suppress this warning."
        elseif warn && haskey_check && !isdata_check
            @warn "`predictvars` object has `$(variable)` specification, but model has no **data** variable named `$(variable)`. The `$(variable)` specification is ignored. Use `warn = false` to suppress this warning."
        end
        return haskey_check && isdata_check
    end

    # Second, for each random variable and predicting variable entry we create an actor
    actors_rv = Dict(variable => make_actor(vardict[variable], value) for (variable, value) in pairs(returnvars) if __check_has_randomvar(vardict, variable))
    actors_pr = Dict(variable => make_actor(vardict[variable], value) for (variable, value) in pairs(predictvars) if __check_has_prediction(vardict, variable))

    # At third, for each variable entry we create a boolean flag to track their updates
    updates = Dict(variable => MarginalHasBeenUpdated(false) for (variable, _) in pairs(merge(actors_rv, actors_pr)))

    _iterations = something(iterations, 1)
    _iterations isa Integer || error("`iterations` argument must be of type Integer or `nothing`")
    _iterations > 0 || error("`iterations` arguments must be greater than zero")

    fe_actor = nothing
    fe_subscription = VoidTeardown()

    potential_error = nothing
    executed_iterations = 0

    try
        on_marginal_update = inference_get_callback(callbacks, :on_marginal_update)
        subscriptions_rv   = Dict(variable => subscribe!(obtain_marginal(vardict[variable]) |> ensure_update(fmodel, on_marginal_update, variable, updates[variable]), actor) for (variable, actor) in pairs(actors_rv))
        subscriptions_pr   = Dict(variable => subscribe!(obtain_prediction(vardict[variable]) |> ensure_update(fmodel, on_marginal_update, variable, updates[variable]), actor) for (variable, actor) in pairs(actors_pr))

        if !isempty(actors_pr) && is_free_energy
            error("The Bethe Free Energy computation is not compatible with the prediction functionality. Set `free_energy = false` to suppress this error.")
        end

        if is_free_energy
            fe_actor        = ScoreActor(S, _iterations, 1)
            fe_subscription = subscribe!(score(fmodel, fe_objective, free_energy_diagnostics), fe_actor)
        end

        if isnothing(data) || isempty(data)
            error("Data is empty. Make sure you used `data` keyword argument with correct value.")
        else
            foreach(filter(pair -> isdata(last(pair)) && !isanonymous(last(pair)), pairs(vardict))) do pair
                varname = first(pair)
                haskey(data, varname) || error(
                    "Data entry `$(varname)` is missing in `data` or `predictvars` arguments. Double check `data = ($(varname) = ???, )` or `predictvars = ($(varname) = ???, )`"
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

        progress_meter = showprogress ? ProgressMeter.Progress(_iterations) : nothing
        cacheddatavars = Dict((key => getvariable(vardict[key]) for key in keys(fdata)))

        for iteration in 1:_iterations
            if something(ensure_bool_or_nothing(inference_invoke_callback(callbacks, :before_iteration, fmodel, iteration)), false)::Bool
                break
            end
            inference_invoke_callback(callbacks, :before_data_update, fmodel, data)
            for (key, value) in fdata
                update!(cacheddatavars[key], value)
            end
            inference_invoke_callback(callbacks, :after_data_update, fmodel, data)

            # Check that all requested marginals have been updated and unset the `updated` flag
            # Throws an error if some were not update
            check_and_reset_updated!(updates)

            if !isnothing(progress_meter)
                ProgressMeter.next!(progress_meter)
            end

            executed_iterations += 1

            if something(ensure_bool_or_nothing(inference_invoke_callback(callbacks, :after_iteration, fmodel, iteration)), false)::Bool
                break
            end
        end

        for (_, subscription) in pairs(merge(subscriptions_pr, subscriptions_rv))
            unsubscribe!(subscription)
        end

        inference_invoke_callback(callbacks, :after_inference, fmodel)
    catch error
        potential_error = inference_process_error(error, !catch_exception)
    end

    if !isnothing(fe_actor)
        release!(fe_actor, (_iterations === executed_iterations))
    end

    unsubscribe!(fe_subscription)

    posterior_values = Dict(variable => inference_postprocess(postprocess, getvalues(actor)) for (variable, actor) in pairs(actors_rv))
    predicted_values = Dict(variable => inference_postprocess(postprocess, getvalues(actor)) for (variable, actor) in pairs(actors_pr))
    fe_values        = !isnothing(fe_actor) ? score_snapshot_iterations(fe_actor, executed_iterations) : nothing

    return InferenceResult(posterior_values, predicted_values, fe_values, fmodel, potential_error)
end

function available_callbacks(::typeof(batch_inference))
    return Val((
        :on_marginal_update,
        :before_model_creation,
        :after_model_creation,
        :before_inference,
        :before_iteration,
        :before_data_update,
        :after_data_update,
        :after_iteration,
        :after_inference
    ))
end

function available_events(::typeof(batch_inference))
    return Val(())
end
