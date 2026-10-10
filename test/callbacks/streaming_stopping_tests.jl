@testitem "Streaming early stopping and per-event BFE windows" begin
    using RxInfer, Rocket
    import RxInfer: start, stop

    @model function stopping_stream_model(y)
        # This tree has exact posterior mean y/2 and settles after one data update.
        # Extra iterations therefore test the stopping window, not numerical convergence.
        x ~ Normal(mean = 0.0, variance = 1.0)
        y ~ Normal(mean = x, variance = 1.0)
    end

    function stream_case(
        callbacks; iterations = 8, free_energy = true, postprocess = nothing
    )
        source = Subject(NamedTuple{(:y,), Tuple{Float64}})
        engine = infer(;
            model = stopping_stream_model(),
            datastream = source,
            autoupdates = RxInfer.EmptyAutoUpdateSpecification,
            iterations,
            free_energy,
            keephistory = 3,
            returnvars = (:x,),
            historyvars = (x = KeepEach(),),
            callbacks,
            postprocess,
            autostart = false,
        )
        posteriors = Any[]
        subscription = subscribe!(
            engine.posteriors[:x], q -> push!(posteriors, q)
        )
        start(engine)
        return (; source, engine, posteriors, subscription)
    end

    strategy = StopEarlyIterationStrategy(0.0, 1e-10; window = 4)
    runs = Vector{Float64}[]
    before_spans = Any[]
    state = stream_case((
        before_iteration = e -> push!(before_spans, e.span_id),
        after_iteration = e -> begin
            @test e.span_id == last(before_spans)
            strategy(e)
        end,
        after_inference = e -> push!(runs, copy(strategy.fe_values)),
    ))
    try
        # Identical successive observations catch a missing reset: their BFE is unchanged.
        # A later changed observation also checks that early exit leaves the engine usable.
        for value in (2.0, 2.0, -3.0)
            next!(state.source, (y = value,))
            @test state.engine.is_running
        end
        @test length.(runs) == [4, 4, 4]
        @test length(before_spans) == 12
        @test length(state.posteriors) == 3
        @test mean.(state.posteriors) ≈ [1.0, 1.0, -1.5]
        @test length.(state.engine.history[:x]) == [4, 4, 4]
        @test length(state.engine.free_energy_raw_history) == 12
        @test length(state.engine.free_energy_final_only_history) == 3
        @test state.engine.free_energy_final_only_history ≈ last.(runs)
        @test length(state.engine.free_energy_history) == 4
    finally
        stop(state.engine)
        unsubscribe!(state.subscription)
    end

    # Custom stopping and varying event lengths must not pad or mix BFE frames.
    completed = Int[]
    stage = Ref(0)
    state = stream_case((
        before_inference = e -> (stage[] += 1),
        after_iteration = e -> begin
            push!(completed, e.iteration)
            e.stop_iteration = e.iteration == stage[]
        end,
    ))
    try
        for value in (1.0, 2.0, 3.0, 4.0)
            next!(state.source, (y = value,))
        end
        @test completed == [1, 1, 2, 1, 2, 3, 1, 2, 3, 4]
        @test length(state.posteriors) == 4
        # Only the last three events fit in history: 2 + 3 + 4 actual iterations.
        @test length.(state.engine.history[:x]) == [2, 3, 4]
        @test length(state.engine.free_energy_raw_history) == 9
        @test length(state.engine.free_energy_final_only_history) == 3
        @test length(state.engine.free_energy_history) == 4
    finally
        stop(state.engine)
        unsubscribe!(state.subscription)
    end

    # No callback preserves the full iteration budget.
    state = stream_case(nothing)
    try
        next!(state.source, (y = 2.0,))
        @test length(only(state.engine.history[:x])) == 8
        @test length(state.engine.free_energy_raw_history) == 8
    finally
        stop(state.engine)
        unsubscribe!(state.subscription)
    end

    # Stopping before iteration two must still publish iteration one's posterior.
    state = stream_case((
        before_iteration = e -> (e.stop_iteration = e.iteration == 2),
    ))
    try
        next!(state.source, (y = 2.0,))
        @test length(only(state.engine.history[:x])) == 1
        @test length(state.posteriors) == 1
        @test length(state.engine.free_energy_raw_history) == 1
    finally
        stop(state.engine)
        unsubscribe!(state.subscription)
    end

    state = stream_case((before_iteration = e -> (e.stop_iteration = true),))
    try
        next!(state.source, (y = 2.0,))
        @test isempty(state.posteriors)
        @test isempty(state.engine.history[:x])
        @test isempty(state.engine.free_energy_raw_history)
        @test state.engine.is_running
    finally
        stop(state.engine)
        unsubscribe!(state.subscription)
    end

    # Custom callbacks do not require BFE tracking; the maximum remains enforced.
    completed = Int[]
    state = stream_case(
        (after_iteration = e -> push!(completed, e.iteration),);
        free_energy = false,
        iterations = 3,
    )
    try
        next!(state.source, (y = 2.0,))
        @test completed == [1, 2, 3]
        @test length(only(state.engine.history[:x])) == 3
        @test mean(only(state.posteriors)) ≈ 1.0
    finally
        stop(state.engine)
        unsubscribe!(state.subscription)
    end

    # A reused batch strategy must also start a fresh window for each solve.
    strategy = StopEarlyIterationStrategy(1e-10; window = 4)
    for value in (2.0, -3.0)
        result = infer(;
            model = stopping_stream_model(),
            data = (y = value,),
            iterations = 8,
            free_energy = true,
            returnvars = (x = KeepLast(),),
            callbacks = (after_iteration = strategy,),
        )
        @test length(result.free_energy) == 4
        @test length(strategy.fe_values) == 4
        @test mean(result.posteriors[:x]) ≈ value / 2
    end

    # NoopPostprocess keeps wrappers intact; saved buffers must still be independent.
    state = stream_case(
        (after_iteration = e -> (e.stop_iteration = e.iteration == 2),);
        postprocess = NoopPostprocess(),
    )
    try
        next!(state.source, (y = 2.0,))
        saved = copy(only(state.engine.history[:x]))
        next!(state.source, (y = -3.0,))
        @test first(state.engine.history[:x]) == saved
        @test length.(state.engine.history[:x]) == [2, 2]
        @test mean.(ReactiveMP.getdata.(first(state.engine.history[:x]))) ≈
            [1.0, 1.0]
    finally
        stop(state.engine)
        unsubscribe!(state.subscription)
    end
end
