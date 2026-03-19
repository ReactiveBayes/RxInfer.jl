
@testitem "Test inference trace callbacks" begin
    using RxInfer

    @model function simple_trace_model(y)
        τ ~ Gamma(; shape = 1.0, rate = 1.0)
        y .~ Normal(; mean = 0.0, precision = τ)
    end

    @testset "Basic tracing records events" begin
        trace = RxInferTraceCallbacks()

        result = infer(;
            model = simple_trace_model(),
            data = (y = [1.0, 2.0, 3.0],),
            callbacks = trace,
        )

        events = RxInfer.tracedevents(trace)
        @test !isempty(events)
        @test all(e -> e isa TracedEvent, events)

        event_names = [e.event for e in events]
        @test :before_model_creation in event_names
        @test :after_model_creation in event_names
    end

    @model function trace_iter_model(y)
        τ ~ Gamma(; shape = 1.0, rate = 1.0)
        x ~ Normal(; mean = 0.0, variance = 1.0)
        y .~ Normal(; mean = x, precision = τ)
    end

    @testset "Tracing with iterations records iteration events" begin
        @constraints function trace_iter_constraints()
            q(x, τ) = q(x)q(τ)
        end

        init = @initialization begin
            q(τ) = Gamma(1.0, 1.0)
        end

        trace = RxInferTraceCallbacks()

        result = infer(;
            model = trace_iter_model(),
            data = (y = [1.0, 2.0, 3.0],),
            callbacks = trace,
            iterations = 5,
            initialization = init,
            constraints = trace_iter_constraints(),
        )

        events = RxInfer.tracedevents(trace)
        event_names = [e.event for e in events]

        @test :before_inference in event_names
        @test :after_inference in event_names
        @test :before_iteration in event_names
        @test :after_iteration in event_names

        @test count(==(:before_iteration), event_names) == 5
        @test count(==(:after_iteration), event_names) == 5
    end
end

@testitem "TracedEvents should include events from ReactiveMP" begin
    using RxInfer
    import RxInfer: tracedevents

    @model function simple_coin_model(y)
        t ~ Beta(1, 1)
        y ~ Bernoulli(t)
    end

    result = infer(; model = simple_coin_model(), data = (y = 1,), trace = true)

    @test result.posteriors[:t] == Beta(2, 1)

    trace = result.model.metadata[:trace]

    @test length(tracedevents(:on_marginal_update, trace)) === 1

    @test length(tracedevents(:before_message_rule_call, trace)) === 2
    @test length(tracedevents(:after_message_rule_call, trace)) === 2
    @test length(tracedevents(:before_product_of_two_messages, trace)) === 1
    @test length(tracedevents(:after_product_of_two_messages, trace)) === 1
    @test length(tracedevents(:before_product_of_messages, trace)) === 1
    @test length(tracedevents(:after_product_of_messages, trace)) === 1
    @test length(tracedevents(:before_form_constraint_applied, trace)) === 1
    @test length(tracedevents(:after_form_constraint_applied, trace)) === 1
    @test length(tracedevents(:before_marginal_computation, trace)) === 1
    @test length(tracedevents(:after_marginal_computation, trace)) === 1
end

@testitem "Trace callbacks should be accessible from model metadata" begin
    using RxInfer

    @model function simple_model_for_trace_metadata(y)
        τ ~ Gamma(; shape = 1.0, rate = 1.0)
        y .~ Normal(; mean = 0.0, precision = τ)
    end

    @testset "Result from the regular callbacks application" begin
        trace = RxInferTraceCallbacks()

        result = infer(;
            model = simple_model_for_trace_metadata(),
            data = (y = [1.0, 2.0, 3.0],),
            callbacks = trace,
        )

        @test haskey(result.model.metadata, :trace)
        @test result.model.metadata[:trace] === trace
    end

    @testset "Result from the trace keyword argument to the `infer` function" begin
        result = infer(;
            model = simple_model_for_trace_metadata(),
            data = (y = [1.0, 2.0, 3.0],),
            trace = true,
        )

        @test haskey(result.model.metadata, :trace)
        @test result.model.metadata[:trace] isa RxInferTraceCallbacks
    end

    @testset "Should error when both `trace = true` and `RxInferTraceCallbacks` are provided" begin
        @test_throws "already contains a `:trace` key" infer(;
            model = simple_model_for_trace_metadata(),
            data = (y = [1.0, 2.0, 3.0],),
            callbacks = RxInferTraceCallbacks(),
            trace = true,
        )
    end
end

@testitem "trace = true should be compatible with custom callbacks and StopIteration" begin
    using RxInfer

    @model function simple_model_for_trace_and_stop(y)
        τ ~ Gamma(; shape = 1.0, rate = 1.0)
        y .~ Normal(; mean = 0.0, precision = τ)
    end

    max_iterations = 50
    stopped_at = Ref(0)

    result = infer(;
        model = simple_model_for_trace_and_stop(),
        data = (y = [1.0, 2.0, 3.0],),
        iterations = max_iterations,
        trace = true,
        callbacks = (
            after_iteration = (model, iteration) -> begin
                stopped_at[] = iteration
                if iteration >= 3
                    return StopIteration()
                end
                return nothing
            end,
        ),
    )

    @test stopped_at[] == 3
    @test haskey(result.model.metadata, :trace)
    trace = result.model.metadata[:trace]
    @test trace isa RxInferTraceCallbacks

    events = RxInfer.tracedevents(trace)
    event_names = [e.event for e in events]
    @test count(==(:before_iteration), event_names) == 3
    @test count(==(:after_iteration), event_names) == 3
end

@testitem "trace = true should be compatible with benchmark = true" begin
    using RxInfer

    @model function simple_model_for_trace_and_benchmark(y)
        τ ~ Gamma(; shape = 1.0, rate = 1.0)
        y .~ Normal(; mean = 0.0, precision = τ)
    end

    result = infer(;
        model = simple_model_for_trace_and_benchmark(),
        data = (y = [1.0, 2.0, 3.0],),
        trace = true,
        benchmark = true,
    )

    @test haskey(result.model.metadata, :trace)
    @test haskey(result.model.metadata, :benchmark)
    @test result.model.metadata[:trace] isa RxInferTraceCallbacks
    @test result.model.metadata[:benchmark] isa RxInferBenchmarkCallbacks
end

@testitem "TracedEvent structure" begin
    using RxInfer

    te = TracedEvent(:test_event, (1, 2, 3))
    @test te.event === :test_event
    @test te.arguments === (1, 2, 3)

    # Test show
    buf = IOBuffer()
    show(buf, te)
    @test String(take!(buf)) == "TracedEvent(:test_event, 3 args)"
end
