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

@testitem "InverseGamma posterior emits shape/scale scalar tags" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    # Normal–InverseGamma model: σ² is variance with a GammaInverse prior, so
    # its posterior arrives as `InverseGamma` (alias of `Distributions.InverseGamma`).
    @model function iid_invgamma(y)
        μ  ~ Normal(mean = 0.0, variance = 100.0)
        σ² ~ GammaInverse(α = 2.0, θ = 1.0)
        y .~ Normal(mean = μ, variance = σ²)
    end

    constraints = @constraints begin
        q(μ, σ²) = q(μ)q(σ²)
    end

    initialization = @initialization begin
        q(μ)  = vague(NormalMeanVariance)
        q(σ²) = vague(GammaInverse)
    end

    dataset = rand(StableRNG(42), NormalMeanVariance(3.1415, 1.0 / 2.7182), 25)

    n_iterations = 4
    results = infer(;
        model          = iid_invgamma(),
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

        # σ² is InverseGamma → shape and scale scalar tags.
        @test "posteriors/σ²/shape" in all_tags
        @test "posteriors/σ²/scale" in all_tags

        # μ stays in the existing Normal family branch.
        @test "posteriors/μ/mean" in all_tags
        @test "posteriors/μ/precision" in all_tags

        # One step per iteration on σ²/shape.
        @test length(steps_for_tag(run_dir, "posteriors/σ²/shape")) ==
            n_iterations
    end
end

@testitem "InverseGamma posterior distribution logged as HistogramSummary" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    @model function iid_invgamma(y)
        μ  ~ Normal(mean = 0.0, variance = 100.0)
        σ² ~ GammaInverse(α = 2.0, θ = 1.0)
        y .~ Normal(mean = μ, variance = σ²)
    end

    constraints = @constraints begin
        q(μ, σ²) = q(μ)q(σ²)
    end

    initialization = @initialization begin
        q(μ)  = vague(NormalMeanVariance)
        q(σ²) = vague(GammaInverse)
    end

    dataset = rand(StableRNG(42), NormalMeanVariance(3.1415, 1.0 / 2.7182), 25)

    n_iterations = 3
    results = infer(;
        model          = iid_invgamma(),
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
            n_samples = 256,
            verbose = false,
        )

        all_tags = read_tags(run_dir)

        @test "posteriors/σ²/distribution" in all_tags
        @test length(steps_for_tag(run_dir, "posteriors/σ²/distribution")) ==
            n_iterations
        @test "posteriors/σ²/shape" in all_tags
        @test "posteriors/σ²/scale" in all_tags
    end
end

@testitem "Poisson posterior emits rate scalar tag" begin
    using RxInfer, TensorBoardLogger
    using Distributions: Poisson
    include(joinpath(@__DIR__, "helpers.jl"))

    # Poisson posteriors arrive in RxInfer as predictive marginals on
    # unobserved Poisson children (see ReactiveMP `@rule Poisson(:out, ...)`
    # under a Gamma rate). Engineering a full predictive model just to
    # exercise the scalar dispatch would be model-specific churn, so we call
    # the dispatch helper directly via the loaded extension module — same
    # pattern as the generic-fallback test below.
    ext = Base.get_extension(RxInfer, :TensorBoardLoggerExt)
    @test ext !== nothing

    with_safe_tempdir() do log_dir
        logger = TBLogger(log_dir, tb_append)
        ctx = ext.LogContext(
            logger;
            log_distributions = false,
            log_text_events   = false,
            n_samples         = 0,
        )

        # Two updates so we can lock in the per-variable step counter
        # behaviour the existing scalars rely on.
        ext._log_posterior_scalars!(ctx, Poisson(3.0), :n)
        ext._log_posterior_scalars!(ctx, Poisson(4.5), :n)

        close(logger)
        empty!(logger.all_files)
        GC.gc()

        all_tags = read_tags(log_dir)
        @test "posteriors/n/rate" in all_tags

        # Specific Poisson dispatch must beat the generic mean/var fallback.
        @test !("posteriors/n/mean" in all_tags)
        @test !("posteriors/n/var"  in all_tags)

        @test length(steps_for_tag(log_dir, "posteriors/n/rate")) == 2
    end
end

@testitem "Geometric posterior emits succprob scalar tag" begin
    using RxInfer, TensorBoardLogger
    using Distributions: Geometric
    include(joinpath(@__DIR__, "helpers.jl"))

    # ReactiveMP has no native Geometric message rules, so a Geometric
    # posterior would only ever arrive via a custom factor or projection.
    # Drive the dispatch helper directly via the loaded extension module —
    # same pattern as the Poisson and generic-fallback tests.
    ext = Base.get_extension(RxInfer, :TensorBoardLoggerExt)
    @test ext !== nothing

    with_safe_tempdir() do log_dir
        logger = TBLogger(log_dir, tb_append)
        ctx = ext.LogContext(
            logger;
            log_distributions = false,
            log_text_events   = false,
            n_samples         = 0,
        )

        ext._log_posterior_scalars!(ctx, Geometric(0.3), :k)
        ext._log_posterior_scalars!(ctx, Geometric(0.6), :k)

        close(logger)
        empty!(logger.all_files)
        GC.gc()

        all_tags = read_tags(log_dir)
        @test "posteriors/k/succprob" in all_tags

        # Specific Geometric dispatch must beat the generic mean/var fallback.
        @test !("posteriors/k/mean" in all_tags)
        @test !("posteriors/k/var"  in all_tags)

        @test length(steps_for_tag(log_dir, "posteriors/k/succprob")) == 2
    end
end

@testitem "NegativeBinomial posterior emits r/succprob scalar tags" begin
    using RxInfer, TensorBoardLogger
    using Distributions: NegativeBinomial
    include(joinpath(@__DIR__, "helpers.jl"))

    # ReactiveMP has no native NegativeBinomial message rules, so a
    # NegativeBinomial posterior would only ever arrive via a custom
    # factor or projection. Drive the dispatch helper directly via the
    # loaded extension module — same pattern as the Poisson, Geometric,
    # and generic-fallback tests.
    ext = Base.get_extension(RxInfer, :TensorBoardLoggerExt)
    @test ext !== nothing

    with_safe_tempdir() do log_dir
        logger = TBLogger(log_dir, tb_append)
        ctx = ext.LogContext(
            logger;
            log_distributions = false,
            log_text_events   = false,
            n_samples         = 0,
        )

        ext._log_posterior_scalars!(ctx, NegativeBinomial(5.0, 0.4), :k)
        ext._log_posterior_scalars!(ctx, NegativeBinomial(7.0, 0.6), :k)

        close(logger)
        empty!(logger.all_files)
        GC.gc()

        all_tags = read_tags(log_dir)
        @test "posteriors/k/r"        in all_tags
        @test "posteriors/k/succprob" in all_tags

        # Specific NegativeBinomial dispatch must beat the generic fallback.
        @test !("posteriors/k/mean" in all_tags)
        @test !("posteriors/k/var"  in all_tags)

        @test length(steps_for_tag(log_dir, "posteriors/k/r"))        == 2
        @test length(steps_for_tag(log_dir, "posteriors/k/succprob")) == 2
    end
end

@testitem "Binomial posterior emits ntrials/succprob scalar tags" begin
    using RxInfer, TensorBoardLogger
    using Distributions: Binomial
    include(joinpath(@__DIR__, "helpers.jl"))

    # Beta is conjugate to Binomial on the success probability — the posterior
    # over θ is Beta, not Binomial. A Binomial marginal therefore arrives only
    # via projection or as a predictive child. Drive the dispatch helper
    # directly via the loaded extension module.
    ext = Base.get_extension(RxInfer, :TensorBoardLoggerExt)
    @test ext !== nothing

    with_safe_tempdir() do log_dir
        logger = TBLogger(log_dir, tb_append)
        ctx = ext.LogContext(
            logger;
            log_distributions = false,
            log_text_events   = false,
            n_samples         = 0,
        )

        ext._log_posterior_scalars!(ctx, Binomial(10, 0.4), :k)
        ext._log_posterior_scalars!(ctx, Binomial(20, 0.6), :k)

        close(logger)
        empty!(logger.all_files)
        GC.gc()

        all_tags = read_tags(log_dir)
        @test "posteriors/k/ntrials"  in all_tags
        @test "posteriors/k/succprob" in all_tags

        # Specific Binomial dispatch must beat the generic mean/var fallback.
        @test !("posteriors/k/mean" in all_tags)
        @test !("posteriors/k/var"  in all_tags)

        @test length(steps_for_tag(log_dir, "posteriors/k/ntrials"))  == 2
        @test length(steps_for_tag(log_dir, "posteriors/k/succprob")) == 2
    end
end

@testitem "Beta-Binomial conjugate workflow emits Binomial predictive tags" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    using Distributions: Binomial
    using Statistics: mean
    include(joinpath(@__DIR__, "helpers.jl"))

    # End-to-end Beta–Binomial: run real RxInfer inference under a Beta
    # prior + Bernoulli likelihood (RxInfer has native rules for those),
    # then turn each iteration's Beta posterior over θ into a Binomial
    # predictive over a count of n_trials future flips and log it via the
    # new Binomial dispatch. This locks in the workflow the dev script in
    # `src/tensorboard_binomial_beta.jl` demonstrates.
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
        returnvars     = (θ = KeepEach(),),
        trace          = true,
    )

    @test length(results.posteriors[:θ]) == n_iterations

    ext = Base.get_extension(RxInfer, :TensorBoardLoggerExt)
    @test ext !== nothing

    with_safe_tempdir() do log_dir
        # Stage 1 — standard trace export to populate the Beta scalars.
        run_dir = RxInfer.convert_to_tensorboard(
            results.model.metadata[:trace];
            output_file       = log_dir,
            log_distributions = false,
            verbose           = false,
        )

        # Stage 2 — append Binomial predictive scalars in the same run dir.
        n_trials = 20
        logger   = TBLogger(run_dir, tb_append)
        ctx      = ext.LogContext(
            logger;
            log_distributions = false,
            log_text_events   = false,
            n_samples         = 0,
        )
        for θ_post in results.posteriors[:θ]
            ext._log_posterior_scalars!(
                ctx, Binomial(n_trials, mean(θ_post)), :y_pred
            )
        end
        close(logger)
        empty!(logger.all_files)
        GC.gc()

        all_tags = read_tags(run_dir)

        # Beta posterior tags from stage 1.
        @test "posteriors/θ/alpha"   in all_tags
        @test "posteriors/θ/beta"    in all_tags

        # Binomial predictive tags from stage 2.
        @test "posteriors/y_pred/ntrials"  in all_tags
        @test "posteriors/y_pred/succprob" in all_tags

        @test length(steps_for_tag(run_dir, "posteriors/y_pred/ntrials")) ==
            n_iterations
    end
end

@testitem "Exponential posterior emits rate scalar tag" begin
    using RxInfer, TensorBoardLogger
    using Distributions: Exponential
    include(joinpath(@__DIR__, "helpers.jl"))

    # Exponential is conjugate-on-rate to Gamma — the posterior over a rate
    # parameter is Gamma, not Exponential. An Exponential marginal therefore
    # arrives only via projection or as a predictive child of a Gamma rate.
    # Drive the dispatch helper directly via the loaded extension module.
    ext = Base.get_extension(RxInfer, :TensorBoardLoggerExt)
    @test ext !== nothing

    with_safe_tempdir() do log_dir
        logger = TBLogger(log_dir, tb_append)
        ctx = ext.LogContext(
            logger;
            log_distributions = false,
            log_text_events   = false,
            n_samples         = 0,
        )

        ext._log_posterior_scalars!(ctx, Exponential(2.0), :x)
        ext._log_posterior_scalars!(ctx, Exponential(0.5), :x)

        close(logger)
        empty!(logger.all_files)
        GC.gc()

        all_tags = read_tags(log_dir)
        @test "posteriors/x/rate" in all_tags

        # Specific Exponential dispatch must beat the generic mean/var fallback.
        @test !("posteriors/x/mean" in all_tags)
        @test !("posteriors/x/var"  in all_tags)

        @test length(steps_for_tag(log_dir, "posteriors/x/rate")) == 2
    end
end

@testitem "VonMises posterior emits location/concentration scalar tags" begin
    using RxInfer, TensorBoardLogger
    using Distributions: VonMises
    include(joinpath(@__DIR__, "helpers.jl"))

    # ReactiveMP has no native VonMises message rules and no VonMises
    # graph node, so a VonMises posterior would only ever arrive via a
    # custom factor or projection. Drive the dispatch helper directly via
    # the loaded extension module — same pattern as Geometric and the
    # other no-rule distributions.
    ext = Base.get_extension(RxInfer, :TensorBoardLoggerExt)
    @test ext !== nothing

    with_safe_tempdir() do log_dir
        logger = TBLogger(log_dir, tb_append)
        ctx = ext.LogContext(
            logger;
            log_distributions = false,
            log_text_events   = false,
            n_samples         = 0,
        )

        ext._log_posterior_scalars!(ctx, VonMises(0.0, 2.0), :θ)
        ext._log_posterior_scalars!(ctx, VonMises(0.5, 5.0), :θ)

        close(logger)
        empty!(logger.all_files)
        GC.gc()

        all_tags = read_tags(log_dir)
        @test "posteriors/θ/location"      in all_tags
        @test "posteriors/θ/concentration" in all_tags

        # Specific VonMises dispatch must beat the generic mean/var fallback.
        @test !("posteriors/θ/mean" in all_tags)
        @test !("posteriors/θ/var"  in all_tags)

        @test length(steps_for_tag(log_dir, "posteriors/θ/location"))      == 2
        @test length(steps_for_tag(log_dir, "posteriors/θ/concentration")) == 2
    end
end

@testitem "Generic UnivariateDistribution fallback emits mean/var" begin
    using RxInfer, TensorBoardLogger
    using Distributions: Uniform
    include(joinpath(@__DIR__, "helpers.jl"))

    # No special-case dispatch exists for `Uniform`, so it should fall
    # through to the generic UnivariateDistribution method emitting `mean`
    # and `var`. Reach into the loaded extension module to call the helper
    # directly — engineering a model whose ReactiveMP posterior arrives as
    # a non-conjugate Distributions.jl type would be model-specific churn.
    ext = Base.get_extension(RxInfer, :TensorBoardLoggerExt)
    @test ext !== nothing

    with_safe_tempdir() do log_dir
        logger = TBLogger(log_dir, tb_append)
        ctx = ext.LogContext(
            logger;
            log_distributions = false,
            log_text_events   = false,
            n_samples         = 0,
        )

        # Uniform(0, 1) — no specific dispatch; falls through to the
        # generic UnivariateDistribution method.
        ext._log_posterior_scalars!(ctx, Uniform(0.0, 1.0), :x)

        close(logger)
        empty!(logger.all_files)
        GC.gc()

        all_tags = read_tags(log_dir)
        @test "posteriors/x/mean" in all_tags
        @test "posteriors/x/var"  in all_tags
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
