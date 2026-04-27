@testitem "Tier A callback events render compact, single-line `Base.show` output" begin
    import ReactiveMP

    # The show methods take any model/engine type — `nameof(typeof(...))` is what
    # ends up in the rendered string, so a plain marker struct keeps the golden
    # output stable without dragging in real `ProbabilisticModel` construction.
    struct FakeModel end
    struct FakeEngine end

    # `OnMarginalUpdateEvent.update` is unwrapped via `ReactiveMP.getdata`. Use a
    # local wrapper so the rendered value is fully deterministic.
    struct PayloadWrapper{T}
        payload::T
    end
    ReactiveMP.getdata(w::PayloadWrapper) = w.payload

    span = "ab12cd34-1234-5678-90ab-cdef12345678"
    model = FakeModel()
    engine = FakeEngine()

    @test repr(BeforeModelCreationEvent(span)) == "BeforeModelCreationEvent(span=ab12…)"
    @test repr(AfterModelCreationEvent(model, span)) ==
        "AfterModelCreationEvent(model=FakeModel, span=ab12…)"

    @test repr(BeforeInferenceEvent(model, span)) ==
        "BeforeInferenceEvent(model=FakeModel, span=ab12…)"
    @test repr(AfterInferenceEvent(model, span)) ==
        "AfterInferenceEvent(model=FakeModel, span=ab12…)"

    @test repr(BeforeIterationEvent(model, 3, span)) ==
        "BeforeIterationEvent(iter=3, span=ab12…)"
    @test repr(AfterIterationEvent(model, 3, span)) ==
        "AfterIterationEvent(iter=3, span=ab12…)"

    # `stop_iteration = true` only surfaces in the rendered output when set.
    stopping = BeforeIterationEvent(model, 3, true, span)
    @test repr(stopping) == "BeforeIterationEvent(iter=3, stop=true, span=ab12…)"

    data_keys = (y = 1, x = 2)  # NamedTuple preserves insertion order in `keys`
    @test repr(BeforeDataUpdateEvent(model, data_keys, span)) ==
        "BeforeDataUpdateEvent(data=[:y, :x], span=ab12…)"
    @test repr(AfterDataUpdateEvent(model, data_keys, span)) ==
        "AfterDataUpdateEvent(data=[:y, :x], span=ab12…)"

    @test repr(OnMarginalUpdateEvent(model, :θ, PayloadWrapper(0.5))) ==
        "OnMarginalUpdateEvent(var=:θ, update=0.5)"

    @test repr(BeforeAutostartEvent(engine, span)) ==
        "BeforeAutostartEvent(engine=FakeEngine, span=ab12…)"
    @test repr(AfterAutostartEvent(engine, span)) ==
        "AfterAutostartEvent(engine=FakeEngine, span=ab12…)"
end

@testitem "_show_span falls back to the full string when shorter than 4 chars" begin
    buf = IOBuffer()
    RxInfer._show_span(buf, "ab")
    @test String(take!(buf)) == "ab"

    buf = IOBuffer()
    RxInfer._show_span(buf, "abcd")
    @test String(take!(buf)) == "abcd…"
end
