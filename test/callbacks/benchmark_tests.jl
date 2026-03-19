
@testitem "Test inference benchmark statistics" begin
    using RxInfer

    callbacks = RxInferBenchmarkCallbacks()

    # A simple model for testing that resembles a simple kalman filter with
    # random walk state transition and unknown observational noise
    @model function test_model1(y)
        τ ~ Gamma(; shape = 1.0, rate = 1.0)

        x[1] ~ Normal(; mean = 0.0, variance = 1.0)
        y[1] ~ Normal(; mean = x[1], precision = τ)

        for i in 2:length(y)
            x[i] ~ Normal(; mean = x[i - 1], variance = 1.0)
            y[i] ~ Normal(; mean = x[i], precision = τ)
        end

        return length(y), 2, 3.0, "hello world" # test returnval
    end

    @constraints function test_model1_constraints()
        q(x, τ) = q(x)q(τ)
    end

    init = @initialization begin
        q(τ) = Gamma(1.0, 1.0)
    end

    infer(;
        model = test_model1(),
        data = (y = [1.0, 2.0, 3.0],),
        callbacks = callbacks,
        iterations = 10,
        initialization = init,
        constraints = test_model1_constraints(),
    )
    @test length(callbacks.before_model_creation_ts) == 1
    @test length(callbacks.after_model_creation_ts) == 1
    @test first(callbacks.before_model_creation_ts) <
        first(callbacks.after_model_creation_ts)
    @test length(callbacks.before_inference_ts) == 1
    @test length(callbacks.after_inference_ts) == 1
    @test first(callbacks.before_inference_ts) <
        first(callbacks.after_inference_ts)
    @test length(callbacks.before_iteration_ts) == 1
    @test length(callbacks.after_iteration_ts) == 1
    @test length(last(callbacks.before_iteration_ts)) == 10
    @test length(last(callbacks.after_iteration_ts)) == 10

    callbacks = RxInferBenchmarkCallbacks()
    for i in 1:10
        infer(;
            model = test_model1(),
            data = (y = [1.0, 2.0, 3.0],),
            callbacks = callbacks,
            iterations = 10,
            initialization = init,
            constraints = test_model1_constraints(),
        )
        @test length(callbacks.before_model_creation_ts) == i
        @test length(callbacks.after_model_creation_ts) == i
        @test last(callbacks.before_model_creation_ts) <
            last(callbacks.after_model_creation_ts)
        @test length(callbacks.before_inference_ts) == i
        @test length(callbacks.after_inference_ts) == i
        @test last(callbacks.before_inference_ts) <
            last(callbacks.after_inference_ts)
        @test length(callbacks.before_iteration_ts) == i
        @test length(callbacks.after_iteration_ts) == i
        length(last(callbacks.before_iteration_ts)) == 10
        @test length(last(callbacks.after_iteration_ts)) == 10
    end

    stats = RxInfer.get_benchmark_stats(callbacks)
    for line in eachrow(stats)
        @test line[2] > 0.0
        @test line[3] > line[2]
        @test line[2] < line[4] < line[3]
        @test line[2] < line[5] < line[3]
        @test !isnan(line[6])
    end

    @model function kalman_filter(x_prev_mean, x_prev_var, τ_shape, τ_rate, y)
        x_prev ~ Normal(; mean = x_prev_mean, variance = x_prev_var)
        τ ~ Gamma(; shape = τ_shape, rate = τ_rate)

        # Random walk with fixed precision
        x_current ~ Normal(; mean = x_prev, precision = 1.0)
        y ~ Normal(; mean = x_current, precision = τ)
    end

    # We assume the following factorisation between variables 
    # in the variational distribution
    @constraints function filter_constraints()
        q(x_prev, x_current, τ) = q(x_prev, x_current)q(τ)
    end
    static_observations = rand(300)
    callbacks           = RxInferBenchmarkCallbacks()
    datastream          = from(static_observations) |> map(NamedTuple{(:y,), Tuple{Float64}}, (d) -> (y = d,))
    autoupdates         = @autoupdates begin
        x_prev_mean, x_prev_var = mean_var(q(x_current))
        τ_shape = shape(q(τ))
        τ_rate = rate(q(τ))
    end

    init = @initialization begin
        q(x_current) = NormalMeanVariance(0.0, 1e3)
        q(τ) = GammaShapeRate(1.0, 1.0)
    end

    engine = infer(;
        model          = kalman_filter(),
        constraints    = filter_constraints(),
        datastream     = datastream,
        autoupdates    = autoupdates,
        returnvars     = (:x_current,),
        keephistory    = 10_000,
        historyvars    = (x_current = KeepLast(), τ = KeepLast()),
        initialization = init,
        iterations     = 10,
        free_energy    = true,
        autostart      = true,
        callbacks      = callbacks,
    )

    @test length(callbacks.before_model_creation_ts) == 1
    @test length(callbacks.after_model_creation_ts) == 1
    @test length(callbacks.before_autostart_ts) == 1
    @test length(callbacks.after_autostart_ts) == 1
end

@testitem "Benchmark callbacks should be accessible from model metadata" begin
    using RxInfer

    @model function simple_model_for_benchmark_metadata(y)
        τ ~ Gamma(; shape = 1.0, rate = 1.0)
        y .~ Normal(; mean = 0.0, precision = τ)
    end

    @testset "Result from the regular callbacks application" begin
        callbacks = RxInferBenchmarkCallbacks()

        result = infer(;
            model = simple_model_for_benchmark_metadata(),
            data = (y = [1.0, 2.0, 3.0],),
            callbacks = callbacks,
        )

        @test haskey(result.model.metadata, :benchmark)
        @test result.model.metadata[:benchmark] === callbacks
    end

    @testset "Result from the benchmark keyword argument to the `infer` function" begin
        result = infer(;
            model = simple_model_for_benchmark_metadata(),
            data = (y = [1.0, 2.0, 3.0],),
            benchmark = true,
        )

        @test haskey(result.model.metadata, :benchmark)
    end
end
