
@testitem "It should be possible to assign event handlers for ReactiveMP #1" begin
    @model function simple_model_for_reactivemp_callbacks(y)
        t ~ Beta(2, 3)
        y ~ Bernoulli(t)
    end

    events = []

    callbacks = (
        before_message_rule_call = (args...) ->
            push!(events, (event = :before_message_rule_call, args = args)),
        after_message_rule_call = (args...) ->
            push!(events, (event = :after_message_rule_call, args = args)),
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
        e -> e.event === :before_message_rule_call, events
    )
    after_message_rule_call_events = filter(
        e -> e.event === :after_message_rule_call, events
    )

    # The model is simple enough, the prior message from `Beta` shouldn't trigger the 
    # re-computation on the second iteration, but only computed once on the first iteration
    # 1st iteration: (m from Beta to t) + (m from Bernoulli to t)
    # 2st iteration: (m from Bernoulli to t)
    @test length(before_message_rule_call_events) == 3
    @test length(after_message_rule_call_events) == 3

    # so in the first iteration we don't really know if message from Beta comes first or second
    # but we know, that it should be either, and not both
    if after_message_rule_call_events[1].args[4] === Beta(2, 3)
        @test after_message_rule_call_events[1].args[4] === Beta(2, 3)
        @test after_message_rule_call_events[2].args[4] === Beta(2, 1)
    else
        @test after_message_rule_call_events[2].args[4] === Beta(2, 3)
        @test after_message_rule_call_events[1].args[4] === Beta(2, 1)
    end

    # we also know exactly what the third message should be
    @test after_message_rule_call_events[3].args[4] === Beta(2, 1)
end

@testitem "It should be possible to write arbitrary information in the model's context from callbacks" begin
    @model function simple_model_for_writing_to_extras_from_callbacks(y)
        t ~ Beta(2, 3)
        y ~ Bernoulli(t)
    end

    struct ArbitraryCallbackHandler end

    function ReactiveMP.invoke_callback(
        ::ArbitraryCallbackHandler, event, args...
    )
        return nothing # ignore all other events, except for the ones below
    end

    function ReactiveMP.invoke_callback(
        ::ArbitraryCallbackHandler,
        event::Val{:on_marginal_update},
        model,
        name,
        update,
    )
        saved_context = get!(() -> [], model.metadata, :saved_context)
        push!(saved_context, (name, RxInfer.ReactiveMP.getdata(update)))
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
