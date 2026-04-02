@testitem "Perfetto export" begin
    using RxInfer
    using JSON

    @model function perfetto_test_model(y)
        τ ~ Gamma(; shape = 1.0, rate = 1.0)
        y .~ Normal(; mean = 0.0, precision = τ)
    end

    traces = let
        results = infer(;
            model = perfetto_test_model(),
            data = (y = [1.0, 2.0, 3.0],),
            trace = true,
        )
        trace = results.model.metadata[:trace]
        RxInfer.tracedevents(trace)
    end

    @testset "_traces_to_perfetto_json produces valid JSON" begin
        json_str = RxInfer._traces_to_perfetto_json(traces)
        parsed = JSON.parse(json_str)
        @test haskey(parsed, "traceEvents")
        @test haskey(parsed, "metadata")
        @test parsed["traceEvents"] isa Vector
        @test !isempty(parsed["traceEvents"])
    end

    @testset "Perfetto events have correct structure" begin
        json_str = RxInfer._traces_to_perfetto_json(traces)
        parsed = JSON.parse(json_str)

        events = parsed["traceEvents"]
        for ev in events
            @test haskey(ev, "pid")
            @test haskey(ev, "tid")
            @test haskey(ev, "name")
            @test haskey(ev, "ph")
            @test haskey(ev, "ts")
            @test ev["ph"] in ("B", "E", "X")
            @test ev["ts"] isa Number
        end
    end

    
    @testset "perfetto_view renders as text/html" begin
        result = perfetto_view(traces)
        html = sprint() do io
            show(io, MIME"text/html"(), result)
        end
        @test occursin("ui.perfetto.dev", html)
        @test occursin("<iframe", html)
        
        # The HTML should contain the Perfetto iframe
        @test occursin("ui.perfetto.dev", html)
        @test occursin("base64", html) || occursin("buffer", html)
    end
end
