# Rocket is used only at the public data/posterior boundary. The factor graph,
# messages, products, local marginals and schedule are the compact backend's.
struct CompiledRecent{P, S}
    program::P
    slots::S
end
Rocket.getrecent(recent::CompiledRecent) =
    compiled_snapshot(recent.program, recent.slots)

struct CompiledObservationTarget{P, N}
    program::P
    nodes::N
end
function ReactiveMP.new_observation!(target::CompiledObservationTarget, value)
    compiled_update_targets!(
        target.program, target.nodes, __normalize_data_indexing(get_data(value))
    )
end
compiled_update_targets!(program, node::GraphPPL.NodeData, value) =
    compiled_update_observation!(node, program, value)
function compiled_update_targets!(program, nodes::AbstractArray, value)
    for I in CartesianIndices(size(nodes))
        isassigned(nodes, I.I...) || continue
        compiled_update_targets!(program, nodes[I.I...], value[I])
    end
end

function compiled_autoupdate_ref(label, index, variables)
    haskey(variables, label) ||
        throw(ArgumentError("Autoupdates references unknown variable $label"))
    return isempty(index) ? variables[label] : variables[label][index...]
end
compiled_map_refs(f, ref::GraphVariableRef) = f(ref)
compiled_map_refs(f, refs::AbstractArray) = _map_sparse_or_dense(f, refs)

function compiled_prepare_autoupdates(specification, builder, roots)
    graph, program = getmodel(builder.model), builder.program
    variables = GraphPPL.variables(getvardict(builder.model))
    target(label::AutoUpdateVariableLabel) = CompiledObservationTarget(
        program,
        compiled_map_refs(
            ref -> graph[getlabel(ref)],
            compiled_autoupdate_ref(
                getlabel(label), getindex(label), variables
            ),
        ),
    )
    target(labels::Tuple) = map(target, labels)
    function argument(
        arg::Union{
            AutoUpdateFetchMarginalArgument, AutoUpdateFetchMessageArgument
        },
    )
        refs = compiled_autoupdate_ref(getlabel(arg), getindex(arg), variables)
        slots = compiled_map_refs(refs) do ref
            label = getlabel(ref)
            node = graph[label]
            slot = if arg isa AutoUpdateFetchMarginalArgument
                getextra(node, CompiledMarginalKey)
            elseif is_random(getproperties(node))
                last(builder.outgoing[label.global_counter])
            else
                getextra(node, CompiledObservationKey)
            end
            push!(roots, slot)
            return slot
        end
        return FetchRecentArgument(
            getlabel(arg), CompiledRecent(program, slots)
        )
    end
    argument(arg::AutoUpdateMapping) =
        AutoUpdateMapping(getmappingfn(arg), map(argument, getarguments(arg)))
    argument(arg) = arg
    return map(specification) do spec
        IndividualAutoUpdateSpecification(
            target(getvarlabels(spec)), argument(getmapping(spec))
        )
    end
end

struct CompiledStreamingState{A, S, H, C, T, D}
    autoupdates::A
    slots::S
    historyvars::H
    callbacks::C
    score_type::Type{T}
    score_slots::Vector{Int32}
    point_entropy_count::Int
    diagnostics::D
end
const CompiledInferenceEngine = RxInferenceEngine{
    T, D, L, V, P, H, S, U, A
} where {T, D, L, V, P, H, S, U, A <: CompiledStreamingState}

function compiled_streaming_inference(;
    model,
    data,
    datastream,
    initialization,
    autoupdates,
    constraints,
    meta,
    options,
    returnvars,
    historyvars,
    keephistory,
    iterations,
    free_energy,
    free_energy_diagnostics,
    allow_node_contraction,
    autostart,
    events,
    annotations,
    callbacks,
    postprocess,
    uselock,
    warn,
)
    allow_node_contraction &&
        throw(UnsupportedCompiledFeature("node contraction"))
    haskey(options, :stream_postprocessors) &&
        throw(UnsupportedCompiledFeature("stream postprocessors"))
    stream, T = if datastream === nothing && data !== nothing
        infer_check_dicttype(:data, data)
        names, items = Tuple(keys(data)), Tuple(values(data))
        labeled(Val(names), iterable(zip(items...))),
        NamedTuple{names, Tuple{eltype.(items)...}}
    else
        eltype(datastream) <: NamedTuple ||
            throw(ArgumentError("datastream must emit NamedTuples"))
        datastream, eltype(datastream)
    end
    config = convert(
        ReactiveMPInferenceOptions,
        (;
            (
                k => v for
                (k, v) in pairs(options) if k ∉ (:runner, :limit_stack_depth)
            )...
        ),
    )
    callbacks = callbacks === nothing ? getcallbacks(config) : callbacks
    config = setcallbacks(config, callbacks)
    annotations === nothing || (config = setannotations(config, annotations))
    postprocess = something(
        postprocess,
        if getannotations(config) === nothing
            UnpackMarginalPostprocess()
        else
            NoopPostprocess()
        end,
    )
    iterations = something(iterations, 1)
    iterations isa Union{Integer, Ref{<:Integer}} && iterations[] > 0 ||
        throw(ArgumentError("iterations must be a positive integer or Ref"))
    keephistory = something(keephistory, 0)
    keephistory isa Integer && keephistory >= 0 ||
        throw(ArgumentError("keephistory must be nonnegative"))
    enabled = something(events, Val(()))
    enabled isa Val &&
    unval(enabled) isa Tuple &&
    all(x -> x isa Symbol, unval(enabled)) ||
        throw(ArgumentError("events must be Val of a tuple of symbols"))
    specification = something(autoupdates, EmptyAutoUpdateSpecification)
    check_model_generator_compatibility(specification, model)
    handlers = autoupdates_data_handlers(specification)
    names = fieldnames(T)
    isempty(intersect(names, keys(handlers))) || throw(
        ArgumentError("Data and autoupdates cannot update the same variable"),
    )
    condition = merge_data_handlers(
        create_deferred_data_handlers(names), handlers
    )
    is_fe, S = unwrap_free_energy_option(free_energy)
    builder, creation_span = compiled_model_builder(
        model,
        condition,
        initialization,
        constraints,
        meta,
        config,
        options.runner,
        is_fe ? __as_counting_real_type(S) : nothing,
    )
    fmodel, program = builder.model, builder.program
    roots = copy(builder.score_slots)
    prepared = compiled_prepare_autoupdates(specification, builder, roots)
    compiled_lower_variables!(builder; required = roots)
    append!(roots, builder.score_slots)
    vardict = GraphPPL.variables(getvardict(fmodel))
    returnvars = something(
        returnvars,
        collect(keys(compiled_selection(fmodel, KeepLast(), false, nothing))),
    )
    all(x -> x isa Symbol, returnvars) ||
        throw(ArgumentError("returnvars must contain symbols"))
    returnselection = Dict(name => KeepLast() for name in returnvars)
    if keephistory > 0
        historyvars = if historyvars === nothing
            Dict(
                name => (iterations[] > 1 ? KeepEach() : KeepLast()) for
                name in returnvars
            )
        else
            compiled_selection(fmodel, historyvars, false, iterations[])
        end
    else
        historyvars === nothing ||
            !warn ||
            @warn "historyvars requires keephistory > 0; ignoring historyvars"
        historyvars = Dict{Symbol, Any}()
    end
    slots = compiled_selected_slots(
        fmodel, merge(returnselection, historyvars), false
    )
    for selected in values(slots)
        if selected isa Int32
            push!(roots, selected)
        else
            append!(roots, vec(selected))
        end
    end
    compiled_finalize!(builder, roots)
    invoke_callback(callbacks, AfterModelCreationEvent(fmodel, creation_span))
    state = CompiledStreamingState(
        prepared,
        slots,
        historyvars,
        callbacks,
        S,
        builder.score_slots,
        builder.point_entropy_count,
        free_energy_diagnostics,
    )
    posteriors = Dict(name => Subject(Any) for name in returnvars)
    history = if keephistory > 0
        Dict(name => CircularBuffer(keephistory) for name in keys(historyvars))
    else
        nothing
    end
    fe_actor = if is_fe && keephistory > 0
        ScoreActor(S, iterations[], keephistory)
    else
        nothing
    end
    fe_source = is_fe ? Subject(S) : nothing
    ticklock = if uselock === true
        ReentrantLock()
    elseif uselock === false
        nothing
    else
        uselock
    end
    engine = RxInferenceEngine(
        T,
        stream,
        nothing,
        nothing,
        posteriors,
        nothing,
        history,
        keephistory > 0 ? historyvars : nothing,
        state,
        fe_actor,
        fe_source,
        postprocess,
        iterations,
        fmodel,
        vardict,
        enabled,
        Subject(RxInferenceEvent),
        ticklock,
        warn,
    )
    if autostart
        span = generate_span_id(callbacks)
        invoke_callback(callbacks, BeforeAutostartEvent(engine, span))
        start(engine)
        invoke_callback(callbacks, AfterAutostartEvent(engine, span))
    end
    return engine
end

function start(engine::CompiledInferenceEngine)
    if engine.is_completed || engine.is_errored || engine.is_running
        engine.warn &&
            @warn "Cannot start an exhausted or already-running inference engine"
        return nothing
    end
    inference_fire_event(
        Val(:before_start), Val(engine.enabled_events), engine.events, engine
    )
    engine.is_running = true
    if engine.fe_actor !== nothing
        engine.fe_subscription = subscribe!(engine.fe_source, engine.fe_actor)
    end
    try
        engine.mainsubscription = subscribe!(
            engine.datastream,
            RxInferenceEventExecutor(eltype(engine.datastream), engine),
        )
    catch
        engine.is_running = false
        unsubscribe!(engine.fe_subscription)
        rethrow()
    end
    inference_fire_event(
        Val(:after_start), Val(engine.enabled_events), engine.events, engine
    )
    return nothing
end

function stop(engine::CompiledInferenceEngine)
    rxexecutorlock(engine.ticklock) do
        engine.is_running || return nothing
        inference_fire_event(
            Val(:before_stop), Val(engine.enabled_events), engine.events, engine
        )
        unsubscribe!(engine.mainsubscription)
        unsubscribe!(engine.fe_subscription)
        engine.is_running = false
        inference_fire_event(
            Val(:after_stop), Val(engine.enabled_events), engine.events, engine
        )
    end
    return nothing
end

function Rocket.on_next!(
    executor::RxInferenceEventExecutor{T, E}, event::T
) where {T, E <: CompiledInferenceEngine}
    engine = executor.engine
    engine.is_running || return nothing
    rxexecutorlock(engine.ticklock) do
        state, model = engine.autoupdates, engine.model
        program = model.metadata[:inference_runner]
        fire(name, args...) = inference_fire_event(
            Val(name), Val(engine.enabled_events), engine.events, args...
        )
        try
            engine.iterations[] > 0 ||
                throw(ArgumentError("iterations must remain positive"))
            if engine.fe_actor !== nothing &&
                engine.iterations[] != getniterations(engine.fe_actor)
                throw(
                    UnsupportedCompiledFeature(
                        "changing iterations while retaining rectangular free-energy history",
                    ),
                )
            end
            fire(:on_new_data, model, event)
            specs = getspecifications(state.autoupdates)
            prefetched = map(fetch, specs)
            snapshots = Dict{Symbol, Any}(
                name => (keep isa KeepEach ? Any[] : nothing) for
                (name, keep) in pairs(state.historyvars)
            )
            for iteration in 1:engine.iterations[]
                fire(:before_iteration, model, iteration)
                fire(:before_auto_update, model, iteration, state.autoupdates)
                run_autoupdate!(specs, prefetched)
                fire(:after_auto_update, model, iteration, state.autoupdates)
                fire(:before_data_update, model, iteration, event)
                compiled_update_data!(model, program, event)
                ReactiveMP.compiled_sweep!(program)
                fire(:after_data_update, model, iteration, event)
                for (name, slots) in pairs(state.slots)
                    value = compiled_snapshot(program, slots)
                    compiled_has_result(value) || throw(
                        ArgumentError(
                            "Compiled inference has unresolved dependencies for $name",
                        ),
                    )
                    invoke_callback(
                        state.callbacks,
                        OnMarginalUpdateEvent(model, name, value),
                    )
                    if haskey(state.historyvars, name)
                        processed = deepcopy(
                            inference_postprocess(engine.postprocess, value)
                        )
                        if state.historyvars[name] isa KeepEach
                            push!(snapshots[name], processed)
                        else
                            (snapshots[name] = processed)
                        end
                    end
                end
                if engine.fe_source !== nothing
                    value = compiled_free_energy(
                        program,
                        state.score_slots,
                        state.score_type,
                        state.point_entropy_count,
                        state.diagnostics,
                    )
                    next!(engine.fe_source, value)
                end
                fire(:after_iteration, model, iteration)
            end
            engine.fe_actor === nothing || release!(engine.fe_actor)
            if engine.history !== nothing
                fire(:before_history_save, model)
                for (name, value) in snapshots
                    push!(engine.history[name], value)
                end
                fire(:after_history_save, model)
            end
            for (name, subject) in pairs(engine.posteriors)
                next!(
                    subject,
                    inference_postprocess(
                        engine.postprocess,
                        compiled_snapshot(program, state.slots[name]),
                    ),
                )
            end
            fire(:on_tick, model)
        catch error
            Rocket.on_error!(executor, error)
        end
    end
    return nothing
end

function Rocket.on_error!(
    executor::RxInferenceEventExecutor{T, E}, error
) where {T, E <: CompiledInferenceEngine}
    engine = executor.engine
    engine.is_errored = true
    engine.is_running = false
    engine.error = error
    engine.model.metadata[:inference_runner].failed = true
    unsubscribe!(engine.fe_subscription)
    unsubscribe!(engine.mainsubscription)
    inference_fire_event(
        Val(:on_error),
        Val(engine.enabled_events),
        engine.events,
        engine.model,
        error,
    )
    inference_process_error(error)
end

function Rocket.on_complete!(
    executor::RxInferenceEventExecutor{T, E}
) where {T, E <: CompiledInferenceEngine}
    engine = executor.engine
    engine.is_completed = true
    engine.is_running = false
    unsubscribe!(engine.fe_subscription)
    inference_fire_event(
        Val(:on_complete),
        Val(engine.enabled_events),
        engine.events,
        engine.model,
    )
    return nothing
end
