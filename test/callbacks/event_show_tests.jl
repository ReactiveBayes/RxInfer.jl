@testitem "Tier A callback events render compact `Base.show` output" begin
    import ReactiveMP

    # The show methods take any model/engine type — `nameof(typeof(...))` is what
    # ends up in the rendered string under `:compact => true`, so a plain marker
    # struct keeps the golden output stable without dragging in real
    # `ProbabilisticModel` construction.
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

    # `sprint` with `:compact => true` mirrors what the trace logger does.
    compact(x) = sprint(show, x; context = :compact => true)

    @test compact(BeforeModelCreationEvent(span)) ==
        "BeforeModelCreationEvent(span=ab12…)"
    @test compact(AfterModelCreationEvent(model, span)) ==
        "AfterModelCreationEvent(model=FakeModel, span=ab12…)"

    @test compact(BeforeInferenceEvent(model, span)) ==
        "BeforeInferenceEvent(model=FakeModel, span=ab12…)"
    @test compact(AfterInferenceEvent(model, span)) ==
        "AfterInferenceEvent(model=FakeModel, span=ab12…)"

    @test compact(BeforeIterationEvent(model, 3, span)) ==
        "BeforeIterationEvent(iter=3, span=ab12…)"
    @test compact(AfterIterationEvent(model, 3, span)) ==
        "AfterIterationEvent(iter=3, span=ab12…)"

    # `stop_iteration = true` only surfaces in the rendered output when set.
    stopping = BeforeIterationEvent(model, 3, true, span)
    @test compact(stopping) ==
        "BeforeIterationEvent(iter=3, stop=true, span=ab12…)"

    data_keys = (y = 1, x = 2)  # NamedTuple preserves insertion order in `keys`
    @test compact(BeforeDataUpdateEvent(model, data_keys, span)) ==
        "BeforeDataUpdateEvent(data=[:y, :x], span=ab12…)"
    @test compact(AfterDataUpdateEvent(model, data_keys, span)) ==
        "AfterDataUpdateEvent(data=[:y, :x], span=ab12…)"

    @test compact(OnMarginalUpdateEvent(model, :θ, PayloadWrapper(0.5))) ==
        "OnMarginalUpdateEvent(var=:θ, update=0.5)"

    @test compact(BeforeAutostartEvent(engine, span)) ==
        "BeforeAutostartEvent(engine=FakeEngine, span=ab12…)"
    @test compact(AfterAutostartEvent(engine, span)) ==
        "AfterAutostartEvent(engine=FakeEngine, span=ab12…)"
end

@testitem "Tier A callback events render full `Base.show` output by default" begin
    import ReactiveMP

    struct FakeModel end
    struct FakeEngine end

    struct PayloadWrapper{T}
        payload::T
    end
    ReactiveMP.getdata(w::PayloadWrapper) = w.payload

    span = "ab12cd34-1234-5678-90ab-cdef12345678"
    model = FakeModel()
    engine = FakeEngine()

    # Default form (REPL/Pluto) keeps the full span id and uses `show` on the
    # model/engine, so `repr` no longer collapses to the type name.
    rendered = repr(AfterModelCreationEvent(model, span))
    @test occursin("span_id=" * span, rendered)
    @test !occursin("span=ab12…", rendered)

    # Same for the engine-flavored events.
    @test occursin("span_id=" * span, repr(BeforeAutostartEvent(engine, span)))

    # Iteration events don't carry a model/engine field, so the full form
    # is identical to the compact form except for the span field.
    @test repr(BeforeIterationEvent(model, 3, span)) ==
        "BeforeIterationEvent(iter=3, span_id=" * span * ")"
    @test repr(AfterIterationEvent(model, 3, span)) ==
        "AfterIterationEvent(iter=3, span_id=" * span * ")"
end

@testitem "Tier A callback events omit span field when span_id is nothing" begin
    import ReactiveMP

    struct FakeModel end
    struct FakeEngine end

    model = FakeModel()
    engine = FakeEngine()

    compact(x) = sprint(show, x; context = :compact => true)

    # When callbacks are disabled the span id is `nothing` and the helper
    # must not surface a `span=nothing` field in either form.
    for ev in (
        BeforeModelCreationEvent(nothing),
        AfterModelCreationEvent(model, nothing),
        BeforeInferenceEvent(model, nothing),
        AfterInferenceEvent(model, nothing),
        BeforeIterationEvent(model, 1, nothing),
        AfterIterationEvent(model, 1, nothing),
        BeforeDataUpdateEvent(model, (y = 1,), nothing),
        AfterDataUpdateEvent(model, (y = 1,), nothing),
        BeforeAutostartEvent(engine, nothing),
        AfterAutostartEvent(engine, nothing),
    )
        @test !occursin("span", compact(ev))
        @test !occursin("span", repr(ev))
    end
end

@testitem "_show_span helper handles compact, full, short, and nothing inputs" begin
    # Compact + long input — 4-char prefix.
    buf = IOBuffer()
    RxInfer._show_span(IOContext(buf, :compact => true), "abcd1234")
    @test String(take!(buf)) == ", span=abcd…"

    # Compact + short input — falls back to the literal value.
    buf = IOBuffer()
    RxInfer._show_span(IOContext(buf, :compact => true), "ab")
    @test String(take!(buf)) == ", span=ab"

    # Full form — `span_id=<full>` with the leading separator.
    buf = IOBuffer()
    RxInfer._show_span(buf, "abcd1234")
    @test String(take!(buf)) == ", span_id=abcd1234"

    # `nothing` writes nothing at all.
    buf = IOBuffer()
    RxInfer._show_span(buf, nothing)
    @test String(take!(buf)) == ""

    # `leading_sep = false` for the only-field-on-the-line case.
    buf = IOBuffer()
    RxInfer._show_span(
        IOContext(buf, :compact => true), "abcd1234"; leading_sep = false
    )
    @test String(take!(buf)) == "span=abcd…"
end
