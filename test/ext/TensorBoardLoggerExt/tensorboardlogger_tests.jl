@testitem "IID estimation trace to TensorBoard" begin
    using RxInfer, StableRNGs, TensorBoardLogger

    # A simple IID model: observations are drawn from a Normal with unknown mean and precision.
    # Mean-field constraints decouple q(μ) and q(τ) for variational inference.
    @model function iid_estimation(y)
        μ  ~ Normal(mean = 0.0, precision = 0.1)
        τ  ~ Gamma(shape = 1.0, rate = 1.0)
        y .~ Normal(mean = μ, precision = τ)
    end

    constraints = @constraints begin
        q(μ, τ) = q(μ)q(τ)
    end

    initialization = @initialization begin
        q(μ) = vague(NormalMeanPrecision)
        q(τ) = vague(GammaShapeRate)
    end

    # Generate synthetic observations from a known distribution so the test is reproducible.
    hidden_μ       = 3.1415
    hidden_τ       = 2.7182
    dataset        = rand(StableRNG(42), NormalMeanPrecision(hidden_μ, hidden_τ), 25)

    # Run inference with `trace = true` so all internal events are recorded.
    # The trace is stored in the model metadata under the `:trace` key.
    results = infer(
        model          = iid_estimation(),
        data           = (y = dataset,),
        constraints    = constraints,
        iterations     = 2,
        initialization = initialization,
        trace          = true,
    )

    trace = results.model.metadata[:trace]

    # Export the trace to TensorBoard format and verify the output.
    # `mktempdir` ensures the log files are written to a temporary directory
    # that is cleaned up automatically after the test, avoiding filesystem pollution.
    mktempdir() do log_dir
        output = RxInfer.convert_to_tensorboard(trace; output_file = log_dir)
        @test output == log_dir  # function returns the path it wrote to
        @test isdir(log_dir)     # directory was created
    end
end

@testitem "Posterior mean/precision scalars logged to TensorBoard" begin
    using RxInfer, StableRNGs, TensorBoardLogger

    # Same IID model — μ is univariate Normal, τ is Gamma.
    # Both variables appear under `posteriors/*`, tagged with parameterization-specific names:
    # Normal → mean/precision, Gamma → shape/rate.
    @model function iid_estimation(y)
        μ  ~ Normal(mean = 0.0, precision = 0.1)
        τ  ~ Gamma(shape = 1.0, rate = 1.0)
        y .~ Normal(mean = μ, precision = τ)
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
    results = infer(
        model          = iid_estimation(),
        data           = (y = dataset,),
        constraints    = constraints,
        iterations     = n_iterations,
        initialization = initialization,
        trace          = true,
    )

    trace = results.model.metadata[:trace]

    mktempdir() do log_dir
        RxInfer.convert_to_tensorboard(trace; output_file = log_dir)

        all_tags = TensorBoardLogger.tags(log_dir)

        # μ is univariate Normal → should produce mean and precision scalar tags
        @test "posteriors/μ/mean" in all_tags
        @test "posteriors/μ/precision" in all_tags

        # τ is Gamma → should produce shape and rate scalar tags
        @test "posteriors/τ/shape" in all_tags
        @test "posteriors/τ/rate" in all_tags

        # Verify we emitted one step per iteration for μ's mean
        mean_steps = BitSet()
        TensorBoardLogger.map_summaries(log_dir; tags = ["posteriors/μ/mean"]) do tag, iter, val
            push!(mean_steps, iter)
        end
        @test length(mean_steps) == n_iterations

        # Same guard for τ's shape — one distinct step per iteration
        shape_steps = BitSet()
        TensorBoardLogger.map_summaries(log_dir; tags = ["posteriors/τ/shape"]) do tag, iter, val
            push!(shape_steps, iter)
        end
        @test length(shape_steps) == n_iterations
    end
end

@testitem "Posterior distributions logged as HistogramSummary" begin
    using RxInfer, StableRNGs, TensorBoardLogger

    # Same IID model — μ is univariate Normal, τ is Gamma. With `log_distributions=true`
    # both posteriors should produce a per-iteration HistogramSummary under
    # `posteriors/<var>/distribution`, which TensorBoard renders in both the
    # Distributions (percentile-band) and Histograms (ridgeline) dashboards.
    @model function iid_estimation(y)
        μ  ~ Normal(mean = 0.0, precision = 0.1)
        τ  ~ Gamma(shape = 1.0, rate = 1.0)
        y .~ Normal(mean = μ, precision = τ)
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
    results = infer(
        model          = iid_estimation(),
        data           = (y = dataset,),
        constraints    = constraints,
        iterations     = n_iterations,
        initialization = initialization,
        trace          = true,
    )

    trace = results.model.metadata[:trace]

    mktempdir() do log_dir
        RxInfer.convert_to_tensorboard(trace; output_file = log_dir,
                                              log_distributions = true,
                                              n_bins = 32)

        all_tags = TensorBoardLogger.tags(log_dir)

        # Distribution tags should exist for both the Normal and Gamma posteriors.
        @test "posteriors/μ/distribution" in all_tags
        @test "posteriors/τ/distribution" in all_tags

        # One HistogramSummary per iteration for each variable.
        μ_steps = BitSet()
        TensorBoardLogger.map_summaries(log_dir; tags = ["posteriors/μ/distribution"]) do tag, iter, val
            push!(μ_steps, iter)
        end
        @test length(μ_steps) == n_iterations

        τ_steps = BitSet()
        TensorBoardLogger.map_summaries(log_dir; tags = ["posteriors/τ/distribution"]) do tag, iter, val
            push!(τ_steps, iter)
        end
        @test length(τ_steps) == n_iterations

        # Scalar tags must still be present — distributions complement, not replace, scalars.
        @test "posteriors/μ/mean" in all_tags
        @test "posteriors/τ/shape" in all_tags
    end
end

@testitem "Default trace export does not emit distribution tags" begin
    using RxInfer, StableRNGs, TensorBoardLogger

    # Guard-rail: with the default `log_distributions=false`, the new code path must be
    # inert — no `posteriors/*/distribution` tags should appear in the log.
    @model function iid_estimation(y)
        μ  ~ Normal(mean = 0.0, precision = 0.1)
        τ  ~ Gamma(shape = 1.0, rate = 1.0)
        y .~ Normal(mean = μ, precision = τ)
    end

    constraints = @constraints begin
        q(μ, τ) = q(μ)q(τ)
    end

    initialization = @initialization begin
        q(μ) = vague(NormalMeanPrecision)
        q(τ) = vague(GammaShapeRate)
    end

    dataset = rand(StableRNG(42), NormalMeanPrecision(3.1415, 2.7182), 10)

    results = infer(
        model          = iid_estimation(),
        data           = (y = dataset,),
        constraints    = constraints,
        iterations     = 2,
        initialization = initialization,
        trace          = true,
    )

    trace = results.model.metadata[:trace]

    mktempdir() do log_dir
        RxInfer.convert_to_tensorboard(trace; output_file = log_dir)
        all_tags = TensorBoardLogger.tags(log_dir)
        @test !("posteriors/μ/distribution" in all_tags)
        @test !("posteriors/τ/distribution" in all_tags)
    end
end
