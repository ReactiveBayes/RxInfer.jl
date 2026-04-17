
@testitem "It should be possible to assign event handlers for ReactiveMP #1" begin
    @model function simple_model_for_reactivemp_callbacks(y)
        t ~ Beta(2, 3)
        y ~ Bernoulli(t)
    end

    events = []

    callbacks = (
        before_message_rule_call = (event) -> push!(
            events, (event_name = :before_message_rule_call, event = event)
        ),
        after_message_rule_call = (event) -> push!(
            events, (event_name = :after_message_rule_call, event = event)
        ),
    )

    result = infer(;
        model = simple_model_for_reactivemp_callbacks(),
        data = (y = 1,),
        callbacks = callbacks,
        iterations = 2,
    )

    @test result.posteriors[:t] == [Beta(3, 3), Beta(3, 3)]
    @test length(events) != 0

    before_message_rule_call_events = filter(
        e -> e.event_name === :before_message_rule_call, events
    )
    after_message_rule_call_events = filter(
        e -> e.event_name === :after_message_rule_call, events
    )

    # The model is simple enough, the prior message from `Beta` shouldn't trigger the
    # re-computation on the second iteration, but only computed once on the first iteration
    # 1st iteration: (m from Beta to t) + (m from Bernoulli to t)
    # 2st iteration: (m from Bernoulli to t)
    @test length(before_message_rule_call_events) == 3
    @test length(after_message_rule_call_events) == 3

    # so in the first iteration we don't really know if message from Beta comes first or second
    # but we know, that it should be either, and not both
    if after_message_rule_call_events[1].event.result === Beta(2, 3)
        @test after_message_rule_call_events[1].event.result === Beta(2, 3)
        @test after_message_rule_call_events[2].event.result === Beta(2, 1)
    else
        @test after_message_rule_call_events[2].event.result === Beta(2, 3)
        @test after_message_rule_call_events[1].event.result === Beta(2, 1)
    end

    # we also know exactly what the third message should be
    @test after_message_rule_call_events[3].event.result === Beta(2, 1)
end

@testitem "It should be possible to write arbitrary information in the model's context from callbacks" begin
    @model function simple_model_for_writing_to_extras_from_callbacks(y)
        t ~ Beta(2, 3)
        y ~ Bernoulli(t)
    end

    struct ArbitraryCallbackHandler end

    # Catch-all: ignore unhandled events
    function ReactiveMP.handle_event(
        ::ArbitraryCallbackHandler, ::ReactiveMP.Event
    )
        return nothing
    end

    function ReactiveMP.handle_event(
        ::ArbitraryCallbackHandler, event::OnMarginalUpdateEvent
    )
        saved_context = get!(() -> [], event.model.metadata, :saved_context)
        push!(
            saved_context,
            (event.variable_name, RxInfer.ReactiveMP.getdata(event.update)),
        )
    end

    result = infer(;
        model = simple_model_for_writing_to_extras_from_callbacks(),
        data = (y = 1,),
        callbacks = ArbitraryCallbackHandler(),
        iterations = 2,
    )

    model = result.model

    @test haskey(model.metadata, :saved_context)
    @test length(model.metadata[:saved_context]) === 2
    @test model.metadata[:saved_context][1] == (:t, Beta(3, 3))
    @test model.metadata[:saved_context][2] == (:t, Beta(3, 3))
end

@testitem "Dispatching on concrete event type vs Event{:name} should produce identical results" begin
    @model function simple_model_for_dispatch_comparison(y)
        t ~ Beta(2, 3)
        y ~ Bernoulli(t)
    end

    # Handler that dispatches on the concrete event type name
    struct ConcreteTypeHandler
        log::Vector{Tuple{Symbol, Int}}
    end

    function ReactiveMP.handle_event(::ConcreteTypeHandler, ::ReactiveMP.Event)
        return nothing
    end

    function ReactiveMP.handle_event(
        handler::ConcreteTypeHandler, event::BeforeIterationEvent
    )
        push!(handler.log, (:before_iteration, event.iteration))
    end

    function ReactiveMP.handle_event(
        handler::ConcreteTypeHandler, event::AfterIterationEvent
    )
        push!(handler.log, (:after_iteration, event.iteration))
    end

    # Handler that dispatches on Event{:name} — should behave identically
    struct EventNameHandler
        log::Vector{Tuple{Symbol, Int}}
    end

    function ReactiveMP.handle_event(::EventNameHandler, ::ReactiveMP.Event)
        return nothing
    end

    function ReactiveMP.handle_event(
        handler::EventNameHandler, event::ReactiveMP.Event{:before_iteration}
    )
        push!(handler.log, (:before_iteration, event.iteration))
    end

    function ReactiveMP.handle_event(
        handler::EventNameHandler, event::ReactiveMP.Event{:after_iteration}
    )
        push!(handler.log, (:after_iteration, event.iteration))
    end

    concrete_handler = ConcreteTypeHandler(Tuple{Symbol, Int}[])
    name_handler = EventNameHandler(Tuple{Symbol, Int}[])

    result1 = infer(;
        model = simple_model_for_dispatch_comparison(),
        data = (y = 1,),
        callbacks = concrete_handler,
        iterations = 3,
    )

    result2 = infer(;
        model = simple_model_for_dispatch_comparison(),
        data = (y = 1,),
        callbacks = name_handler,
        iterations = 3,
    )

    @test result1.posteriors[:t] == result2.posteriors[:t]
    @test concrete_handler.log == name_handler.log
    @test length(concrete_handler.log) == 6  # 3 before + 3 after
    @test concrete_handler.log == [
        (:before_iteration, 1),
        (:after_iteration, 1),
        (:before_iteration, 2),
        (:after_iteration, 2),
        (:before_iteration, 3),
        (:after_iteration, 3),
    ]
end
