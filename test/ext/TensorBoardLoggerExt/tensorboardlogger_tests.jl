@testitem "IID estimation trace to TensorBoard" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    # A simple IID model: observations are drawn from a Normal with unknown mean and precision.
    # Mean-field constraints decouple q(μ) and q(τ) for variational inference.
    @model function iid_estimation(y)
        μ ~ Normal(; mean = 0.0, precision = 0.1)
        τ ~ Gamma(; shape = 1.0, rate = 1.0)
        y .~ Normal(; mean = μ, precision = τ)
    end

    constraints = @constraints begin
        q(μ, τ) = q(μ)q(τ)
    end

    initialization = @initialization begin
        q(μ) = vague(NormalMeanPrecision)
        q(τ) = vague(GammaShapeRate)
    end

    # Generate synthetic observations from a known distribution so the test is reproducible.
    hidden_μ = 3.1415
    hidden_τ = 2.7182
    dataset = rand(StableRNG(42), NormalMeanPrecision(hidden_μ, hidden_τ), 25)

    # Run inference with `trace = true` so all internal events are recorded.
    # The trace is stored in the model metadata under the `:trace` key.
    results = infer(;
        model          = iid_estimation(),
        data           = (y = dataset,),
        constraints    = constraints,
        iterations     = 2,
        initialization = initialization,
        trace          = true,
    )

    trace = results.model.metadata[:trace]

    # Export the trace to TensorBoard format and verify the output.
    # `with_safe_tempdir` keeps the log files in a temp directory that gets
    # retry-cleaned after the test — avoiding Windows EBUSY on `.tfevents`
    # handles that TB readers leave mapped past `close`.
    with_safe_tempdir() do log_dir
        output = RxInfer.convert_to_tensorboard(
            trace; output_file = log_dir, verbose = false
        )
        @test startswith(output, log_dir)  # returned path is a subdirectory of log_dir
        @test isdir(output)                # timestamped subdirectory was created
    end
end

@testitem "Posterior mean/precision scalars logged to TensorBoard" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    # Same IID model — μ is univariate Normal, τ is Gamma.
    # Both variables appear under `posteriors/*`, tagged with parameterization-specific names:
    # Normal → mean/precision, Gamma → shape/rate.
    @model function iid_estimation(y)
        μ ~ Normal(; mean = 0.0, precision = 0.1)
        τ ~ Gamma(; shape = 1.0, rate = 1.0)
        y .~ Normal(; mean = μ, precision = τ)
    end

    constraints = @constraints begin
        q(μ, τ) = q(μ)q(τ)
    end

    initialization = @initialization begin
        q(μ) = vague(NormalMeanPrecision)
        q(τ) = vague(GammaShapeRate)
    end

    dataset = rand(StableRNG(42), NormalMeanPrecision(3.1415, 2.7182), 25)

    n_iterations = 5
    results = infer(;
        model          = iid_estimation(),
        data           = (y = dataset,),
        constraints    = constraints,
        iterations     = n_iterations,
        initialization = initialization,
        trace          = true,
    )

    trace = results.model.metadata[:trace]

    with_safe_tempdir() do log_dir
        run_dir = RxInfer.convert_to_tensorboard(
            trace; output_file = log_dir, verbose = false
        )

        all_tags = read_tags(run_dir)

        # μ is univariate Normal → should produce mean and precision scalar tags
        @test "posteriors/μ/mean" in all_tags
        @test "posteriors/μ/precision" in all_tags

        # τ is Gamma → should produce shape and rate scalar tags
        @test "posteriors/τ/shape" in all_tags
        @test "posteriors/τ/rate" in all_tags

        # Verify we emitted one step per iteration for μ's mean and τ's shape.
        @test length(steps_for_tag(run_dir, "posteriors/μ/mean")) ==
            n_iterations
        @test length(steps_for_tag(run_dir, "posteriors/τ/shape")) ==
            n_iterations
    end
end

@testitem "Posterior distributions logged as HistogramSummary" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    # Same IID model — μ is univariate Normal, τ is Gamma. With `log_distributions=true`
    # both posteriors should produce a per-iteration HistogramSummary under
    # `posteriors/<var>/distribution`, which TensorBoard renders in both the
    # Distributions (percentile-band) and Histograms (ridgeline) dashboards.
    @model function iid_estimation(y)
        μ ~ Normal(; mean = 0.0, precision = 0.1)
        τ ~ Gamma(; shape = 1.0, rate = 1.0)
        y .~ Normal(; mean = μ, precision = τ)
    end

    constraints = @constraints begin
        q(μ, τ) = q(μ)q(τ)
    end

    initialization = @initialization begin
        q(μ) = vague(NormalMeanPrecision)
        q(τ) = vague(GammaShapeRate)
    end

    dataset = rand(StableRNG(42), NormalMeanPrecision(3.1415, 2.7182), 25)

    n_iterations = 4
    results = infer(;
        model          = iid_estimation(),
        data           = (y = dataset,),
        constraints    = constraints,
        iterations     = n_iterations,
        initialization = initialization,
        trace          = true,
    )

    trace = results.model.metadata[:trace]

    with_safe_tempdir() do log_dir
        run_dir = RxInfer.convert_to_tensorboard(
            trace;
            output_file = log_dir,
            log_distributions = true,
            n_samples = 512,
            verbose = false,
        )

        all_tags = read_tags(run_dir)

        # Distribution tags should exist for both the Normal and Gamma posteriors.
        @test "posteriors/μ/distribution" in all_tags
        @test "posteriors/τ/distribution" in all_tags

        # One HistogramSummary per iteration for each variable.
        @test length(steps_for_tag(run_dir, "posteriors/μ/distribution")) ==
            n_iterations
        @test length(steps_for_tag(run_dir, "posteriors/τ/distribution")) ==
            n_iterations

        # Scalar tags must still be present — distributions complement, not replace, scalars.
        @test "posteriors/μ/mean" in all_tags
        @test "posteriors/τ/shape" in all_tags
    end
end

@testitem "Beta posterior emits alpha/beta/mean scalar tags" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    # Beta–Bernoulli coin toss. θ has a Beta posterior; the conjugate update
    # produces α/β scalars per iteration.
    @model function coin_toss(y)
        θ ~ Beta(1.0, 1.0)
        y .~ Bernoulli(θ)
    end

    initialization = @initialization begin
        q(θ) = vague(Beta)
    end

    dataset = rand(StableRNG(42), Bernoulli(0.7), 25)

    n_iterations = 4
    results = infer(;
        model          = coin_toss(),
        data           = (y = dataset,),
        iterations     = n_iterations,
        initialization = initialization,
        trace          = true,
    )

    trace = results.model.metadata[:trace]

    with_safe_tempdir() do log_dir
        run_dir = RxInfer.convert_to_tensorboard(
            trace; output_file = log_dir, verbose = false
        )

        all_tags = read_tags(run_dir)

        # θ is Beta → should produce alpha, beta, and mean scalar tags.
        @test "posteriors/θ/alpha" in all_tags
        @test "posteriors/θ/beta" in all_tags
        @test "posteriors/θ/mean" in all_tags

        # One step per iteration on the canonical α tag.
        @test length(steps_for_tag(run_dir, "posteriors/θ/alpha")) ==
            n_iterations
    end
end

@testitem "Beta posterior distribution logged as HistogramSummary" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    # With `log_distributions = true`, the Beta posterior must produce a
    # per-iteration HistogramSummary alongside the α/β scalar tags.
    @model function coin_toss(y)
        θ ~ Beta(1.0, 1.0)
        y .~ Bernoulli(θ)
    end

    initialization = @initialization begin
        q(θ) = vague(Beta)
    end

    dataset = rand(StableRNG(42), Bernoulli(0.7), 25)

    n_iterations = 3
    results = infer(;
        model          = coin_toss(),
        data           = (y = dataset,),
        iterations     = n_iterations,
        initialization = initialization,
        trace          = true,
    )

    trace = results.model.metadata[:trace]

    with_safe_tempdir() do log_dir
        run_dir = RxInfer.convert_to_tensorboard(
            trace;
            output_file = log_dir,
            log_distributions = true,
            n_samples = 256,
            verbose = false,
        )

        all_tags = read_tags(run_dir)

        @test "posteriors/θ/distribution" in all_tags
        @test length(steps_for_tag(run_dir, "posteriors/θ/distribution")) ==
            n_iterations

        # Scalars must still coexist with the histogram tag.
        @test "posteriors/θ/alpha" in all_tags
        @test "posteriors/θ/beta" in all_tags
    end
end

@testitem "Default trace export does not emit distribution tags" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    # Guard-rail: with the default `log_distributions=false`, the new code path must be
    # inert — no `posteriors/*/distribution` tags should appear in the log.
    @model function iid_estimation(y)
        μ ~ Normal(; mean = 0.0, precision = 0.1)
        τ ~ Gamma(; shape = 1.0, rate = 1.0)
        y .~ Normal(; mean = μ, precision = τ)
    end

    constraints = @constraints begin
        q(μ, τ) = q(μ)q(τ)
    end

    initialization = @initialization begin
        q(μ) = vague(NormalMeanPrecision)
        q(τ) = vague(GammaShapeRate)
    end

    dataset = rand(StableRNG(42), NormalMeanPrecision(3.1415, 2.7182), 10)

    results = infer(;
        model          = iid_estimation(),
        data           = (y = dataset,),
        constraints    = constraints,
        iterations     = 2,
        initialization = initialization,
        trace          = true,
    )

    trace = results.model.metadata[:trace]

    with_safe_tempdir() do log_dir
        run_dir = RxInfer.convert_to_tensorboard(
            trace; output_file = log_dir, verbose = false
        )
        all_tags = read_tags(run_dir)
        @test !("posteriors/μ/distribution" in all_tags)
        @test !("posteriors/τ/distribution" in all_tags)
    end
end

@testitem "Event text logging is opt-in via log_text_events" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    # `log_text_events` gates every per-event text breadcrumb. With the default
    # `false`, none of the narrative tags (`Events`, `before_iteration`, …,
    # `EventCounts`) should appear. Flipping it to `true` must reinstate them
    # without disturbing scalar outputs (`iteration_time_ms`, `posteriors/*/*`).
    @model function iid_estimation(y)
        μ ~ Normal(; mean = 0.0, precision = 0.1)
        τ ~ Gamma(; shape = 1.0, rate = 1.0)
        y .~ Normal(; mean = μ, precision = τ)
    end

    constraints = @constraints begin
        q(μ, τ) = q(μ)q(τ)
    end

    initialization = @initialization begin
        q(μ) = vague(NormalMeanPrecision)
        q(τ) = vague(GammaShapeRate)
    end

    dataset = rand(StableRNG(42), NormalMeanPrecision(3.1415, 2.7182), 10)

    results = infer(;
        model          = iid_estimation(),
        data           = (y = dataset,),
        constraints    = constraints,
        iterations     = 2,
        initialization = initialization,
        trace          = true,
    )

    trace = results.model.metadata[:trace]

    # Default: per-event text breadcrumbs are suppressed; scalars still flow.
    # `EventCounts` is always emitted as a compact run summary, regardless of the flag.
    with_safe_tempdir() do log_dir
        run_dir = RxInfer.convert_to_tensorboard(
            trace; output_file = log_dir, verbose = false
        )
        all_tags = read_tags(run_dir)
        @test !("Events" in all_tags)
        @test !("before_iteration" in all_tags)
        @test !("after_iteration" in all_tags)
        @test "EventCounts" in all_tags
        @test "iteration_time_ms" in all_tags
        @test "posteriors/μ/mean" in all_tags
    end

    # Opt-in: the full narrative layer comes back.
    with_safe_tempdir() do log_dir
        run_dir = RxInfer.convert_to_tensorboard(
            trace;
            output_file = log_dir,
            log_text_events = true,
            verbose = false,
        )
        all_tags = read_tags(run_dir)
        @test "Events" in all_tags
        @test "EventCounts" in all_tags
        @test "before_iteration" in all_tags
        @test "after_iteration" in all_tags
        # Scalars continue to coexist with the text tags.
        @test "iteration_time_ms" in all_tags
        @test "posteriors/μ/mean" in all_tags
    end
end
