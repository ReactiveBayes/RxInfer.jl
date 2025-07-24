import Static

"""
    RxInferenceEngine

The return value of the `infer` function in case of streamlined inference. 

# Public fields
- `posteriors`: `Dict` or `NamedTuple` of 'random variable' - 'posterior stream' pairs. See the `returnvars` argument for the [`infer`](@ref).
- `free_energy`: (optional) A stream of Bethe Free Energy values per VMP iteration. See the `free_energy` argument for the [`infer`](@ref).
- `history`: (optional) Saves history of previous marginal updates. See the `historyvars` and `keephistory` arguments for the [`infer`](@ref).
- `free_energy_history`: (optional) Free energy history, averaged across variational iterations value for all observations  
- `free_energy_raw_history`: (optional) Free energy history, returns returns computed values of all variational iterations for each data event (if available)
- `free_energy_final_only_history`: (optional) Free energy history, returns computed values of final variational iteration for each data event (if available)
- `events`: (optional) A stream of events send by the inference engine. See the `events` argument for the [`infer`](@ref).
- `model`: `ProbabilisticModel` object reference.

Use the `RxInfer.start(engine)` function to subscribe on the `datastream` source and start the inference procedure. 
Use `RxInfer.stop(engine)` to unsubscribe from the `datastream` source and stop the inference procedure. 
Note, that it is not always possible to start/stop the inference procedure.

See also: [`infer`](@ref), [`RxInferenceEvent`](@ref), [`RxInfer.start`](@ref), [`RxInfer.stop`](@ref)
"""
mutable struct RxInferenceEngine{T, D, L, V, P, H, S, U, A, FA, FS, R, I, M, N, X, E, J}
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
    fe_source       :: FS
    fe_subscription :: Teardown

    # utility 
    postprocess  :: R
    iterations   :: I
    model        :: M
    vardict      :: N
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
        fe_source::FS,
        postprocess::R,
        iterations::I,
        model::M,
        vardict::N,
        enabledevents::Val{X},
        events::E,
        ticklock::J
    ) where {T, D, L, V, P, H, S, U, A, FA, FS, R, I, M, N, X, E, J} = begin
        return new{T, D, L, V, P, H, S, U, A, FA, FS, R, I, M, N, X, E, J}(
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
            fe_source,
            voidTeardown,
            postprocess,
            iterations,
            model,
            vardict,
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
    if !isnothing(getfield(engine, :historyactors))
        print(io, "available for (")
        join(io, keys(getfield(engine, :historyactors)), ", ")
        print(io, ")\n")
    else
        print(io, "unavailable\n")
    end

    print(io, rpad("  Free Energy history", lcolumnlen), " | ")
    if !isnothing(getfield(engine, :fe_actor))
        print(io, "available\n")
    else
        print(io, "unavailable\n")
    end

    print(io, rpad("  Enabled events", lcolumnlen), " | ")
    print(io, "[ ", join(enabled_events(engine), ", "), " ]")
end

enabled_events(::RxInferenceEngine{T, D, L, V, P, H, S, U, A, FA, FS, R, I, M, N, X, E}) where {T, D, L, V, P, H, S, U, A, FA, FS, R, I, M, N, X, E} = X

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
            return subscribe!(obtain_marginal(engine.vardict[name]), updateflag)
        end

        if !isnothing(engine.historyactors) && !isnothing(engine.history)
            engine.historysubscriptions = map(keys(engine.historyactors), values(engine.historyactors)) do name, actor
                return subscribe!(obtain_marginal(engine.vardict[name]), actor)
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
        factor_nodes(getmodel(engine.model)) do _, node
            marginal_stream = getextra(node, ReactiveMPExtraMarginalStreamKey, nothing)
            if !isnothing(marginal_stream)
                unsubscribe!(marginal_stream)
            end
        end

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

Base.show(io::IO, ::RxInferenceEventExecutor) = print(io, "RxInferenceEventExecutor")

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
        _postprocess    = executor.engine.postprocess
        _model          = executor.engine.model
        _datavars       = executor.engine.datavars
        _autoupdates    = executor.engine.autoupdates
        _updateflags    = executor.engine.updateflags
        _history        = executor.engine.history
        _historyactors  = executor.engine.historyactors
        _fe_actor       = executor.engine.fe_actor
        _enabled_events = executor.engine.enabled_events
        _events         = executor.engine.events

        inference_invoke_event(Val(:on_new_data), Val(_enabled_events), _events, _model, event)

        # Before we start our iterations we 'prefetch' recent values for autoupdates
        # This is important, because the values linked to the `autoupdate` may (and most likely will) 
        # change during the iterationd
        autoupdate_specs = getspecifications(_autoupdates)
        autoupdate_fetched = map(fetch, autoupdate_specs)

        # This loop correspond to the different VMP iterations
        # Here `_iterations` can be `Ref` too, so we use `[]`. Should not affect integers
        for iteration in 1:_iterations[]
            inference_invoke_event(Val(:before_iteration), Val(_enabled_events), _events, _model, iteration)

            # At first we update all our priors (auto updates) with the fixed values from the `redirectupdate` field
            inference_invoke_event(Val(:before_auto_update), Val(_enabled_events), _events, _model, iteration, _autoupdates)
            run_autoupdate!(autoupdate_specs, autoupdate_fetched)
            inference_invoke_event(Val(:after_auto_update), Val(_enabled_events), _events, _model, iteration, _autoupdates)

            # At second we pass our observations
            inference_invoke_event(Val(:before_data_update), Val(_enabled_events), _events, _model, iteration, event)
            for (datavar, value) in zip(_datavars, values(event))
                update!(datavar, value)
            end
            inference_invoke_event(Val(:after_data_update), Val(_enabled_events), _events, _model, iteration, event)

            check_and_reset_updated!(_updateflags)

            inference_invoke_event(Val(:after_iteration), Val(_enabled_events), _events, _model, iteration)
        end

        # `release!` on `fe_actor` ensures that free energy sumed up between iterations correctly
        if !isnothing(_fe_actor)
            release!(_fe_actor)
        end

        if !isnothing(_history) && !isnothing(_historyactors)
            inference_invoke_event(Val(:before_history_save), Val(_enabled_events), _events, _model)
            for (name, actor) in pairs(_historyactors)
                push!(_history[name], inference_postprocess(_postprocess, getvalues(actor)))
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

    inference_process_error(err)
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

"""
    RxInferenceEvent{T, D}

The `RxInferenceEngine` sends events in a form of the `RxInferenceEvent` structure. `T` represents the type of an event, `D` represents the type of a data associated with the event.
The type of data depends on the type of an event, but usually represents a tuple, which can be unrolled automatically with the Julia's splitting syntax, e.g. `model, iteration = event`. 
See the documentation of the `rxinference` function for possible event types and their associated data types.

The events system itself uses the `Rocket.jl` library API. For example, one may create a custom event listener in the following way:


```julia
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
engine = infer(events = Val((:after_iteration, )), ...)

subscription = subscribe!(engine.events, MyEventListener(...))
```

See also: [`infer`](@ref), [`RxInferenceEngine`](@ref)
"""
struct RxInferenceEvent{T, D}
    data::D

    RxInferenceEvent(::Val{T}, data::D) where {T, D} = new{T, D}(data)
end

event_name(::RxInferenceEvent{T}) where {T} = T

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

function streaming_inference(;
    model::ModelGenerator,
    data = nothing,
    datastream = nothing,
    initialization = nothing,
    autoupdates = nothing,
    constraints = nothing,
    meta = nothing,
    options = nothing,
    returnvars = nothing,
    historyvars = nothing,
    keephistory = nothing,
    iterations = nothing,
    free_energy = false,
    free_energy_diagnostics = DefaultObjectiveDiagnosticChecks,
    allow_node_contraction = false,
    autostart = true,
    events = nothing,
    addons = nothing,
    callbacks = nothing,
    postprocess = DefaultPostprocess(),
    uselock = false,
    warn = true
)

    # In case if `data` is used we cast to a synchronous `datastream` with zip operator
    _datastream, _T = if isnothing(datastream) && !isnothing(data)
        infer_check_dicttype(:data, data)

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

    if getforce_marginal_computation(_options)
        modelplugins = modelplugins + ReactiveMPForceMarginalComputationPlugin(MarginalComputationOptions())
    end

    # The `_model` here still must be a `ModelGenerator`
    _model = GraphPPL.with_backend(GraphPPL.with_plugins(model, modelplugins), ReactiveMPGraphPPLBackend(Static.static(allow_node_contraction)))
    _autoupdates = something(autoupdates, EmptyAutoUpdateSpecification)

    check_model_generator_compatibility(_autoupdates, _model)

    # For each data entry and autoupdate we create a `DeferredDataHandler` handler for the `condition_on` structure 
    # We must do that because the data is not available at the moment of the model creation
    _autoupdates_data_handlers = autoupdates_data_handlers(autoupdates)
    foreach(keys(_autoupdates_data_handlers)) do _autoupdate_data_handler_key
        if _autoupdate_data_handler_key ∈ datavarnames
            error(lazy"`$(_autoupdate_data_handler_key)` is present both in the `data` and in the `autoupdates`.")
        end
    end
    _condition_on = merge_data_handlers(create_deferred_data_handlers(datavarnames), autoupdates_data_handlers(autoupdates))

    inference_invoke_callback(callbacks, :before_model_creation)
    fmodel = create_model(_model | _condition_on)
    inference_invoke_callback(callbacks, :after_model_creation, fmodel)

    vardict = getvardict(fmodel)
    vardict = GraphPPL.variables(vardict) # TODO: Should work recursively as well

    _autoupdates = prepare_autoupdates_for_model(_autoupdates, fmodel)

    # At the very beginning we try to preallocate handles for the `datavar` labels that are present in the `T` (from `datastream`)
    # This is not very type-stable-friendly but we do it once and it should pay-off in the inference procedure
    datavars = ntuple(N) do i
        datavarname = datavarnames[i]
        (haskey(vardict, datavarname) && is_data(vardict[datavarname])) ||
            error("The `datastream` produces data for `$(datavarname)`, but the model does not have a datavar named `$(datavarname)`")
        return getvariable(vardict[datavarname])
    end

    # `iterations` might be set to `nothing` in which case we assume `1` iteration
    _iterations = something(iterations, 1)
    (_iterations isa Integer || _iterations isa Ref{<:Integer}) || error("`iterations` argument must be of type Integer, Ref{<:Integer}, or `nothing`")
    _iterations[] > 0 || error("`iterations` arguments must be greater than zero")

    _keephistory = something(keephistory, 0)
    _keephistory isa Integer || error("`keephistory` argument must be of type Integer or `nothing`")
    _keephistory >= 0 || error("`keephistory` arguments must be greater than or equal to zero")

    # `tickscheduler` defines a moment when we send new posteriors in the `posteriors` streams
    tickscheduler = PendingScheduler()

    # Here we prepare our free energy streams (if requested)
    fe_actor  = nothing
    fe_source = nothing

    if is_free_energy
        if _keephistory > 0
            fe_actor = ScoreActor(S, _iterations[], _keephistory)
        end
        fe_source = score(fmodel, fe_objective, free_energy_diagnostics)
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
        returnvars = [variable for (variable, value) in pairs(vardict) if (israndom(value))]
    end

    eltype(returnvars) === Symbol || error("`returnvars` must contain a list of symbols") # TODO?

    returnvars = filter((varkey) -> __check_has_randomvar(:returnvars, vardict, varkey), returnvars)

    inference_check_itertype(:returnvars, returnvars)

    # `rxinference` by default does not keep track of marginals updates history
    # If user specifies `keephistory` keyword argument
    if _keephistory > 0
        if isnothing(historyvars)
            # First what we do - we check if `historyvars` is nothing 
            # In which case we mirror the `returnvars` specication and use either `KeepLast()` or `KeepEach` (depending on the iterations spec)
            historyoption = _iterations[] > 1 ? KeepEach() : KeepLast()
            historyvars   = Dict(name => historyoption for name in returnvars)
        elseif historyvars === KeepEach() || historyvars === KeepLast()
            # Second we check if it is one of the two possible global values: `KeepEach` and `KeepLast`. 
            # If so, we replace it with either `KeepEach` or `KeepLast` for each random and not-proxied variable in a model
            historyvars = Dict(variable => historyvars for (variable, value) in pairs(vardict) if (israndom(value) && !isanonymous(value)))
        end

        historyvars = Dict((varkey => value) for (varkey, value) in pairs(historyvars) if __check_has_randomvar(:historyvars, vardict, varkey))

        infer_check_dicttype(:historyvars, historyvars)
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
        historyactors = Dict(name => make_actor(vardict[name], historyoption, _iterations[]) for (name, historyoption) in pairs(historyvars))
        history       = Dict(name => CircularBuffer(_keephistory) for (name, _) in pairs(historyvars))
    end

    # At this point we must have properly defined and fixed `returnvars` and `historyvars` objects

    # For each random variable entry in `returnvars` specification we create a boolean flag to track their updates
    updateflags = Dict(variable => MarginalHasBeenUpdated(false) for variable in returnvars)

    # `posteriors` returns a `stream` for each entry in the `returnvars`
    posteriors = Dict(
        variable => obtain_marginal(vardict[variable]) |> schedule_on(tickscheduler) |> map(Any, (data) -> inference_postprocess(postprocess, data)) for variable in returnvars
    )

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
        fe_source,
        postprocess,
        _iterations,
        fmodel,
        vardict,
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

function available_events(::typeof(streaming_inference))
    return Val((
        :before_start,
        :after_start,
        :before_stop,
        :after_stop,
        :on_new_data,
        :before_iteration,
        :before_auto_update,
        :after_auto_update,
        :before_data_update,
        :after_data_update,
        :after_iteration,
        :before_history_save,
        :after_history_save,
        :on_tick,
        :on_error,
        :on_complete
    ))
end

function available_callbacks(::typeof(streaming_inference))
    return Val((:before_model_creation, :after_model_creation, :before_autostart, :after_autostart))
end
