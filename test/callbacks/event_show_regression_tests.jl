@testitem "ReactiveMP callback events render without raw struct dumps" begin
    # Regression for #638: when TBLogger writes per-event Text-tag entries via
    # `_log_text!(ctx, tag, repr(ev); step=idx)`, the rendered `repr(ev)` must
    # not contain raw `MessageMapping{...}(...)` struct dumps. ReactiveMP's
    # compact `Base.show(::IO, ::MessageMapping)` is the fix; this test would
    # regress if the helpers ever fell back to the default struct printer.
    #
    # `Message{` is the analogous telltale for the `Message` type. ReactiveMP
    # only ships `Base.show` for `DeferredMessage` today, so a non-deferred
    # `Message` still falls through to the default printer. The check below
    # is intentionally limited to `MessageMapping{`; once ReactiveMP grows a
    # general `Base.show(::IO, ::Message)`, expand the assertion here.

    @model function tiny_for_event_show_regression(y)
        t ~ Beta(2, 3)
        y ~ Bernoulli(t)
    end

    captured = Any[]
    push_ev = ev -> push!(captured, ev)

    callbacks = (
        before_message_rule_call       = push_ev,
        after_message_rule_call        = push_ev,
        before_product_of_two_messages = push_ev,
        after_product_of_two_messages  = push_ev,
        before_product_of_messages     = push_ev,
        after_product_of_messages      = push_ev,
        before_marginal_computation    = push_ev,
        after_marginal_computation     = push_ev,
    )

    infer(;
        model = tiny_for_event_show_regression(),
        data = (y = 1,),
        callbacks = callbacks,
        iterations = 2,
    )

    @test !isempty(captured)
    for ev in captured
        rendered = repr(ev)
        @test !occursin("MessageMapping{", rendered)
    end
end
