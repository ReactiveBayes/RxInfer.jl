@testitem "Benchmark statistics and session summaries print as tables with PrettyTables" begin
    using RxInfer, PrettyTables

    @model function beta_bernoulli(y)
        θ ~ Beta(1.0, 1.0)
        y .~ Bernoulli(θ)
    end

    callbacks = RxInferBenchmarkCallbacks()
    session = RxInfer.create_session()
    for _ in 1:3
        infer(
            model = beta_bernoulli(),
            data = (y = [1.0, 0.0, 1.0],),
            callbacks = callbacks,
            session = session,
        )
    end

    # Wide enough that no column is cropped
    buffer = IOBuffer()
    io = IOContext(buffer, :displaysize => (50, 200))
    pretty_table(io, callbacks)
    output = String(take!(buffer))
    @test occursin(
        "RxInfer inference benchmark statistics: 3 evaluations", output
    )
    for label in
        ("Operation", "Min", "Max", "Mean", "Median", "Std", "Model creation")
        @test occursin(label, output)
    end
    @test occursin("╭", output) # the rounded unicode borders

    RxInfer.summarize_session(io, session)
    output = String(take!(buffer))
    @test !occursin("PrettyTables.jl is not installed", output)
    for label in ("ID", "Status", "Duration", "Model", "Data", "beta_bernoulli")
        @test occursin(label, output)
    end
end
