@testitem "IID estimation trace to TensorBoard" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    # Smoke test: a real `infer` run with `trace = true` produces a trace that
    # `convert_to_tensorboard` can write to a timestamped subdirectory.
    results = iid_normal_inference()
    trace = results.model.metadata[:trace]

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

    # IID Normal: μ → Normal (mean/precision), τ → Gamma (shape/rate). One
    # step per iteration on each scalar tag.
    n_iterations = 2
    results = iid_normal_inference(iterations = n_iterations)
    trace = results.model.metadata[:trace]

    with_safe_tempdir() do log_dir
        run_dir = RxInfer.convert_to_tensorboard(
            trace; output_file = log_dir, verbose = false
        )
        all_tags = read_tags(run_dir)

        @test "posteriors/μ/mean" in all_tags
        @test "posteriors/μ/precision" in all_tags
        @test "posteriors/τ/shape" in all_tags
        @test "posteriors/τ/rate" in all_tags

        @test length(steps_for_tag(run_dir, "posteriors/μ/mean")) ==
            n_iterations
        @test length(steps_for_tag(run_dir, "posteriors/τ/shape")) ==
            n_iterations
    end
end

@testitem "Posterior distributions logged as HistogramSummary" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    # `log_distributions=true` adds a per-iteration HistogramSummary under
    # `posteriors/<var>/distribution` (rendered in both the Distributions and
    # Histograms dashboards). Scalars must continue to coexist.
    n_iterations = 2
    results = iid_normal_inference(iterations = n_iterations)
    trace = results.model.metadata[:trace]

    with_safe_tempdir() do log_dir
        run_dir = RxInfer.convert_to_tensorboard(
            trace;
            output_file       = log_dir,
            log_distributions = true,
            n_samples         = 64,
            verbose           = false,
        )
        all_tags = read_tags(run_dir)

        @test "posteriors/μ/distribution" in all_tags
        @test "posteriors/τ/distribution" in all_tags
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
    n_iterations = 2
    results = coin_toss_inference(iterations = n_iterations)
    trace = results.model.metadata[:trace]

    with_safe_tempdir() do log_dir
        run_dir = RxInfer.convert_to_tensorboard(
            trace; output_file = log_dir, verbose = false
        )
        all_tags = read_tags(run_dir)

        @test "posteriors/θ/alpha" in all_tags
        @test "posteriors/θ/beta" in all_tags
        @test "posteriors/θ/mean" in all_tags
        @test length(steps_for_tag(run_dir, "posteriors/θ/alpha")) ==
            n_iterations
    end
end

@testitem "Beta posterior distribution logged as HistogramSummary" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    # With `log_distributions=true` the Beta posterior must produce a
    # per-iteration HistogramSummary alongside the α/β scalar tags.
    n_iterations = 2
    results = coin_toss_inference(iterations = n_iterations)
    trace = results.model.metadata[:trace]

    with_safe_tempdir() do log_dir
        run_dir = RxInfer.convert_to_tensorboard(
            trace;
            output_file       = log_dir,
            log_distributions = true,
            n_samples         = 64,
            verbose           = false,
        )
        all_tags = read_tags(run_dir)

        @test "posteriors/θ/distribution" in all_tags
        @test length(steps_for_tag(run_dir, "posteriors/θ/distribution")) ==
            n_iterations
        @test "posteriors/θ/alpha" in all_tags
        @test "posteriors/θ/beta" in all_tags
    end
end

@testitem "InverseGamma posterior emits shape/scale scalar tags" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    # Normal–InverseGamma model: σ² is variance under a GammaInverse prior, so
    # its posterior arrives as `InverseGamma`. μ stays in the Normal branch.
    n_iterations = 2
    results = iid_invgamma_inference(iterations = n_iterations)
    trace = results.model.metadata[:trace]

    with_safe_tempdir() do log_dir
        run_dir = RxInfer.convert_to_tensorboard(
            trace; output_file = log_dir, verbose = false
        )
        all_tags = read_tags(run_dir)

        @test "posteriors/σ²/shape" in all_tags
        @test "posteriors/σ²/scale" in all_tags
        @test "posteriors/μ/mean" in all_tags
        @test "posteriors/μ/precision" in all_tags
        @test length(steps_for_tag(run_dir, "posteriors/σ²/shape")) ==
            n_iterations
    end
end

@testitem "InverseGamma posterior distribution logged as HistogramSummary" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    n_iterations = 2
    results = iid_invgamma_inference(iterations = n_iterations)
    trace = results.model.metadata[:trace]

    with_safe_tempdir() do log_dir
        run_dir = RxInfer.convert_to_tensorboard(
            trace;
            output_file       = log_dir,
            log_distributions = true,
            n_samples         = 64,
            verbose           = false,
        )
        all_tags = read_tags(run_dir)

        @test "posteriors/σ²/distribution" in all_tags
        @test length(steps_for_tag(run_dir, "posteriors/σ²/distribution")) ==
            n_iterations
        @test "posteriors/σ²/shape" in all_tags
        @test "posteriors/σ²/scale" in all_tags
    end
end

@testitem "Univariate posterior scalar dispatch coverage" begin
    using RxInfer, TensorBoardLogger
    using Distributions:
        Poisson,
        Geometric,
        NegativeBinomial,
        Binomial,
        Exponential,
        VonMises,
        Weibull,
        LogNormal,
        Erlang,
        Laplace,
        Pareto,
        Rayleigh,
        Chisq,
        Uniform
    include(joinpath(@__DIR__, "helpers.jl"))

    # Table-driven dispatch coverage. Each case pushes a sequence of events
    # through `_log_posterior_scalars!` and asserts:
    #   * the parameterisation-specific tags appear with one step per event
    #   * the generic UnivariateDistribution fallback is shadowed (`excluded`)
    #
    # ReactiveMP has no native rules for most of these distributions — they
    # arrive only via projection or custom factors — so we drive the dispatch
    # helper directly via the loaded extension module rather than engineering
    # full inference models. The Uniform row is the inverse case: no specific
    # dispatch exists, so it must fall through to the generic mean/var tags.
    cases = [
        (
            label = "Poisson",
            events = [Poisson(3.0), Poisson(4.5)],
            var = :n,
            expected = ["rate"],
            excluded = ["mean", "var"],
        ),
        (
            label = "Geometric",
            events = [Geometric(0.3), Geometric(0.6)],
            var = :k,
            expected = ["succprob"],
            excluded = ["mean", "var"],
        ),
        (
            label = "NegativeBinomial",
            events = [NegativeBinomial(5.0, 0.4), NegativeBinomial(7.0, 0.6)],
            var = :k,
            expected = ["r", "succprob"],
            excluded = ["mean", "var"],
        ),
        (
            label = "Binomial",
            events = [Binomial(10, 0.4), Binomial(20, 0.6)],
            var = :k,
            expected = ["ntrials", "succprob"],
            excluded = ["mean", "var"],
        ),
        (
            label = "Exponential",
            events = [Exponential(2.0), Exponential(0.5)],
            var = :x,
            expected = ["rate"],
            excluded = ["mean", "var"],
        ),
        (
            label = "VonMises",
            events = [VonMises(0.0, 2.0), VonMises(0.5, 5.0)],
            var = :θ,
            expected = ["location", "concentration"],
            excluded = ["mean", "var"],
        ),
        (
            label = "Weibull",
            events = [Weibull(1.5, 2.0), Weibull(2.5, 1.2)],
            var = :t,
            expected = ["shape", "scale"],
            excluded = ["mean", "var"],
        ),
        (
            label = "LogNormal",
            events = [LogNormal(0.0, 1.0), LogNormal(0.5, 0.7)],
            var = :x,
            expected = ["meanlog", "stdlog"],
            excluded = ["mean", "var"],
        ),
        (
            label = "Erlang",
            events = [Erlang(3, 2.0), Erlang(5, 1.5)],
            var = :t,
            expected = ["shape", "scale"],
            excluded = ["mean", "var"],
        ),
        (
            label = "Laplace",
            events = [Laplace(0.0, 1.0), Laplace(0.5, 0.7)],
            var = :x,
            expected = ["location", "scale"],
            excluded = ["mean", "var"],
        ),
        (
            label = "Pareto",
            events = [Pareto(2.5, 1.0), Pareto(3.0, 1.5)],
            var = :x,
            expected = ["shape", "scale"],
            excluded = ["mean", "var"],
        ),
        (
            label = "Rayleigh",
            events = [Rayleigh(1.0), Rayleigh(2.0)],
            var = :r,
            expected = ["scale"],
            excluded = ["mean", "var"],
        ),
        (
            label = "Chisq",
            events = [Chisq(3.0), Chisq(7.0)],
            var = :x,
            expected = ["dof"],
            excluded = ["mean", "var"],
        ),
        (
            label = "Uniform (generic fallback)",
            events = [Uniform(0.0, 1.0)],
            var = :x,
            expected = ["mean", "var"],
            excluded = String[],
        ),
    ]

    @testset "$(case.label)" for case in cases
        result = with_dispatch_logger() do ext, ctx
            for evt in case.events
                ext._log_posterior_scalars!(ctx, evt, case.var)
            end
        end

        for suffix in case.expected
            tag = "posteriors/$(case.var)/$(suffix)"
            @test tag in result.tags
            @test length(get(result.steps, tag, BitSet())) ==
                length(case.events)
        end
        for suffix in case.excluded
            @test !("posteriors/$(case.var)/$(suffix)" in result.tags)
        end
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
    # predictive over `n_trials` future flips and log it via the
    # Binomial dispatch.
    n_iterations = 2
    results = coin_toss_inference(
        iterations = n_iterations, returnvars = (θ = KeepEach(),)
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
        ctx      = ext.LogContext(logger; log_distributions = false, log_text_events = false, n_samples = 0)
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
        @test "posteriors/θ/alpha" in all_tags
        @test "posteriors/θ/beta" in all_tags

        # Binomial predictive tags from stage 2.
        @test "posteriors/y_pred/ntrials" in all_tags
        @test "posteriors/y_pred/succprob" in all_tags

        @test length(steps_for_tag(run_dir, "posteriors/y_pred/ntrials")) ==
            n_iterations
    end
end

@testitem "Default trace export does not emit distribution tags" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    # Guard-rail: with the default `log_distributions=false`, the histogram code
    # path must be inert — no `posteriors/*/distribution` tags should appear.
    results = iid_normal_inference()
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

    # `log_text_events` gates every per-event text breadcrumb. Default `false`
    # suppresses the narrative tags (`Events`, `before_iteration`, …); flipping
    # it to `true` reinstates them without disturbing scalar outputs.
    # `EventCounts` is always emitted as a compact run summary.
    results = iid_normal_inference()
    trace = results.model.metadata[:trace]

    # Default: per-event text breadcrumbs are suppressed; scalars still flow.
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
            output_file     = log_dir,
            log_text_events = true,
            verbose         = false,
        )
        all_tags = read_tags(run_dir)
        @test "Events" in all_tags
        @test "EventCounts" in all_tags
        @test "before_iteration" in all_tags
        @test "after_iteration" in all_tags
        @test "iteration_time_ms" in all_tags
        @test "posteriors/μ/mean" in all_tags
    end
end

@testitem "log_posteriors=false suppresses every posterior tag" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    # `log_posteriors=false` short-circuits both the scalar and histogram
    # paths inside `OnMarginalUpdateEvent`, so no `posteriors/*` tag should
    # appear. Iteration timing is independent and must still flow — proves
    # the gate is scoped to posteriors and not to the whole event loop.
    results = iid_normal_inference()
    trace = results.model.metadata[:trace]

    with_safe_tempdir() do log_dir
        run_dir = RxInfer.convert_to_tensorboard(
            trace;
            output_file       = log_dir,
            log_posteriors    = false,
            log_distributions = true,
            verbose           = false,
        )
        all_tags = read_tags(run_dir)

        # Every posterior tag is suppressed, including the histogram path
        # that `log_distributions=true` would otherwise enable.
        @test !("posteriors/μ/mean" in all_tags)
        @test !("posteriors/μ/precision" in all_tags)
        @test !("posteriors/μ/distribution" in all_tags)
        @test !("posteriors/τ/shape" in all_tags)
        @test !("posteriors/τ/rate" in all_tags)
        @test !("posteriors/τ/distribution" in all_tags)

        # Non-posterior outputs are unaffected.
        @test "iteration_time_ms" in all_tags
        @test "EventCounts" in all_tags
    end
end

@testitem "log_posteriors=true preserves default posterior logging" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    # Regression guard: passing `log_posteriors=true` explicitly must produce
    # the same posterior tags as the historical (always-on) path.
    n_iterations = 2
    results = iid_normal_inference(iterations = n_iterations)
    trace = results.model.metadata[:trace]

    with_safe_tempdir() do log_dir
        run_dir = RxInfer.convert_to_tensorboard(
            trace;
            output_file    = log_dir,
            log_posteriors = true,
            verbose        = false,
        )
        all_tags = read_tags(run_dir)

        @test "posteriors/μ/mean" in all_tags
        @test "posteriors/μ/precision" in all_tags
        @test "posteriors/τ/shape" in all_tags
        @test "posteriors/τ/rate" in all_tags
        @test length(steps_for_tag(run_dir, "posteriors/μ/mean")) ==
            n_iterations
        @test length(steps_for_tag(run_dir, "posteriors/τ/shape")) ==
            n_iterations
    end
end

@testitem "log_posteriors allow-list filters scalar and histogram paths" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    # An allow-list of variable names must restrict logging to only the named
    # marginals across both the scalar and histogram paths. The excluded
    # variable's tags must be entirely absent. We also confirm the allow-list
    # accepts both `String` and `Symbol` element types — the documented
    # dual-input contract — and that an empty allow-list matches `false`.
    n_iterations = 2
    results = iid_normal_inference(iterations = n_iterations)
    trace = results.model.metadata[:trace]

    @testset "String form: log only μ" begin
        with_safe_tempdir() do log_dir
            run_dir = RxInfer.convert_to_tensorboard(
                trace;
                output_file       = log_dir,
                log_posteriors    = ["μ"],
                log_distributions = true,
                verbose           = false,
            )
            all_tags = read_tags(run_dir)

            @test "posteriors/μ/mean" in all_tags
            @test "posteriors/μ/precision" in all_tags
            @test "posteriors/μ/distribution" in all_tags
            @test !("posteriors/τ/shape" in all_tags)
            @test !("posteriors/τ/rate" in all_tags)
            @test !("posteriors/τ/distribution" in all_tags)
            @test length(steps_for_tag(run_dir, "posteriors/μ/mean")) ==
                n_iterations
        end
    end

    @testset "Symbol form: identical filter behaviour" begin
        with_safe_tempdir() do log_dir
            run_dir = RxInfer.convert_to_tensorboard(
                trace;
                output_file    = log_dir,
                log_posteriors = [:μ],
                verbose        = false,
            )
            all_tags = read_tags(run_dir)

            @test "posteriors/μ/mean" in all_tags
            @test "posteriors/μ/precision" in all_tags
            @test !("posteriors/τ/shape" in all_tags)
            @test !("posteriors/τ/rate" in all_tags)
        end
    end

    @testset "Empty allow-list is the moral equivalent of `false`" begin
        with_safe_tempdir() do log_dir
            run_dir = RxInfer.convert_to_tensorboard(
                trace;
                output_file    = log_dir,
                log_posteriors = String[],
                verbose        = false,
            )
            all_tags = read_tags(run_dir)
            @test !("posteriors/μ/mean" in all_tags)
            @test !("posteriors/τ/shape" in all_tags)
            @test "iteration_time_ms" in all_tags
        end
    end
end

@testitem "Summary tag is emitted on real inference runs" begin
    using RxInfer, StableRNGs, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    # End-to-end: a real `infer` call exercises every Tier 1 source — model
    # creation, inference span, and per-iteration timing — so the Summary tag
    # must appear alongside `EventCounts` in the export.
    results = iid_normal_inference()
    trace = results.model.metadata[:trace]

    with_safe_tempdir() do log_dir
        run_dir = RxInfer.convert_to_tensorboard(
            trace; output_file = log_dir, verbose = false
        )
        all_tags = read_tags(run_dir)

        @test "Summary" in all_tags
        @test "EventCounts" in all_tags
        # Summary aggregates iteration timing into a single text snapshot;
        # the per-iteration scalar series must continue to coexist.
        @test "iteration_time_ms" in all_tags
    end
end

@testitem "Summary writer behaviour across context shapes" begin
    using RxInfer, TensorBoardLogger
    include(joinpath(@__DIR__, "helpers.jl"))

    # The Summary writer skips lines whose underlying measurement is missing
    # and skips the entire tag when nothing has been measured. These three
    # subtests drive `_log_summary!` directly on minimally populated contexts
    # to lock in that shape-aware behaviour.
    ext = Base.get_extension(RxInfer, :TensorBoardLoggerExt)
    @test ext !== nothing

    @testset "empty context emits no Summary tag" begin
        # An empty LogContext (no events processed, no iteration durations)
        # must not emit a Summary at all — every line corresponds to a
        # measurement, so a fully empty context has nothing to log. Guards
        # against a misleading "all-zero" Summary in degenerate runs.
        with_safe_tempdir() do log_dir
            logger = TBLogger(log_dir, tb_append)
            ctx = ext.LogContext(
                logger;
                log_distributions = false,
                log_text_events   = false,
                n_samples         = 0,
            )

            ext._log_summary!(ctx)

            close(logger)
            empty!(logger.all_files)
            GC.gc()

            @test !("Summary" in read_tags(log_dir))
        end
    end

    @testset "iteration durations alone produce a Summary" begin
        # Partial coverage (e.g. a streaming/autostart run that bypasses the
        # inference span) should still surface useful timing information when
        # iteration durations are present.
        with_safe_tempdir() do log_dir
            logger = TBLogger(log_dir, tb_append)
            ctx = ext.LogContext(
                logger;
                log_distributions = false,
                log_text_events   = false,
                n_samples         = 0,
            )
            ctx.iteration_durations[1] = 12.5
            ctx.iteration_durations[2] = 8.0
            ctx.iteration_durations[3] = 15.0

            ext._log_summary!(ctx)

            close(logger)
            empty!(logger.all_files)
            GC.gc()

            @test "Summary" in read_tags(log_dir)
        end
    end

    @testset "span timings alone produce a Summary" begin
        # A single-pass run with no variational iterations may still populate
        # the model_build / inference spans — those should produce a Summary
        # table even when iteration_durations is empty.
        with_safe_tempdir() do log_dir
            logger = TBLogger(log_dir, tb_append)
            ctx = ext.LogContext(
                logger;
                log_distributions = false,
                log_text_events   = false,
                n_samples         = 0,
            )
            ctx.model_build_ms = 4.2
            ctx.inference_ms = 17.3
            ctx.first_event_ns = UInt64(1_000_000_000)
            ctx.last_event_ns = UInt64(1_025_000_000)

            ext._log_summary!(ctx)

            close(logger)
            empty!(logger.all_files)
            GC.gc()

            @test "Summary" in read_tags(log_dir)
        end
    end
end
