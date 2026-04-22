
module TensorBoardLoggerExt
    using RxInfer
    using Dates
    using ReactiveMP: event_name, getdata
    using ExponentialFamily: UnivariateNormalDistributionsFamily, GammaDistributionsFamily
    using Distributions: shape, rate
    using Random: MersenneTwister
    using Statistics: mean, var
    using TensorBoardLogger

    # ─── Log context ──────────────────────────────────────────────────────────
    # State holder threaded through every `log_event` method so that per-event
    # dispatch can read/mutate shared counters, iteration timings, and the
    # TBLogger without any top-level `isa` branching in the main loop. Mirrors
    # the `RxInferBenchmarkCallbacks` pattern in `src/callbacks/benchmark.jl`.
    #
    # `current_time_ns` is refreshed by the main loop before each dispatch so
    # that timing-sensitive methods (BeforeIterationEvent / AfterIterationEvent)
    # can read the TracedEvent timestamp without widening every `log_event`
    # method's signature with an unused argument.
    mutable struct LogContext{L}
        logger::L
        iteration_durations::Dict{Int, Float64}
        before_times::Dict{Any, Tuple{Int, UInt64}}
        counts::Dict{Symbol, Int}
        posterior_step::Dict{Symbol, Int}
        log_distributions::Bool
        log_text_events::Bool
        n_samples::Int
        current_time_ns::UInt64
    end

    LogContext(logger; log_distributions::Bool, log_text_events::Bool, n_samples::Int) = LogContext(
        logger,
        Dict{Int, Float64}(),
        Dict{Any, Tuple{Int, UInt64}}(),
        Dict{Symbol, Int}(),
        Dict{Symbol, Int}(),
        log_distributions,
        log_text_events,
        n_samples,
        zero(UInt64),
    )

    # Central gate for all narrative/event text summaries. Scalar and
    # histogram logging stays unconditional — only the per-event text
    # breadcrumbs are opt-in via `log_text_events`.
    @inline _log_text!(ctx::LogContext, tag, msg; step) =
        ctx.log_text_events && TensorBoardLogger.log_text(ctx.logger, tag, msg; step=step)

    # ─── Distribution-family dispatched helpers ──────────────────────────────
    # Deterministic samples via a seeded MersenneTwister keep the HistogramSummary
    # reproducible across re-runs. The `::Any` fallback returns an empty vector
    # so non-univariate or unsupported marginals are silently skipped without
    # any branching at the call site.
    _posterior_samples(dist::UnivariateNormalDistributionsFamily, n::Int) = rand(MersenneTwister(1), dist, n)
    _posterior_samples(dist::GammaDistributionsFamily,            n::Int) = rand(MersenneTwister(1), dist, n)
    _posterior_samples(::Any,                                     ::Int)  = Float64[]

    # Scalar-posterior logging, dispatched on the distribution family's natural
    # parameterisation. Mutates `ctx.posterior_step` so the per-variable step
    # counter stays aligned with the distribution-summary step.
    function _log_posterior_scalars!(ctx::LogContext, dist::UnivariateNormalDistributionsFamily, name::Symbol)
        step = (ctx.posterior_step[name] = get(ctx.posterior_step, name, 0) + 1)
        TensorBoardLogger.log_value(ctx.logger, "posteriors/$(name)/mean",      mean(dist);     step=step)
        TensorBoardLogger.log_value(ctx.logger, "posteriors/$(name)/precision", inv(var(dist)); step=step)
    end
    function _log_posterior_scalars!(ctx::LogContext, dist::GammaDistributionsFamily, name::Symbol)
        step = (ctx.posterior_step[name] = get(ctx.posterior_step, name, 0) + 1)
        TensorBoardLogger.log_value(ctx.logger, "posteriors/$(name)/shape", shape(dist); step=step)
        TensorBoardLogger.log_value(ctx.logger, "posteriors/$(name)/rate",  rate(dist);  step=step)
    end
    _log_posterior_scalars!(::LogContext, ::Any, ::Symbol) = nothing

    # Per-iteration HistogramSummary. The data-only `log_histogram` overload
    # lets TB auto-bin each iteration's samples so HistogramProto.min/max track
    # the actual sample extremes — that is what makes the Distributions plugin
    # narrow the percentile bands as the posterior sharpens.
    function _log_posterior_distribution!(ctx::LogContext, dist::UnivariateNormalDistributionsFamily, name::Symbol)
        samples = _posterior_samples(dist, ctx.n_samples)
        isempty(samples) && return nothing
        step = get(ctx.posterior_step, name, 0)
        TensorBoardLogger.log_histogram(ctx.logger, "posteriors/$(name)/distribution", samples; step=step)
    end
    function _log_posterior_distribution!(ctx::LogContext, dist::GammaDistributionsFamily, name::Symbol)
        samples = _posterior_samples(dist, ctx.n_samples)
        isempty(samples) && return nothing
        step = get(ctx.posterior_step, name, 0)
        TensorBoardLogger.log_histogram(ctx.logger, "posteriors/$(name)/distribution", samples; step=step)
    end
    _log_posterior_distribution!(::LogContext, ::Any, ::Symbol) = nothing

    # ─── Per-event-type logging methods ───────────────────────────────────────

    function log_event(ctx::LogContext, ev::BeforeModelCreationEvent, idx)
        _log_text!(ctx, "before_model_creation",
            "span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(ctx::LogContext, ev::AfterModelCreationEvent, idx)
        _log_text!(ctx, "after_model_creation",
            "model: $(ev.model) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(ctx::LogContext, ev::BeforeInferenceEvent, idx)
        _log_text!(ctx, "before_inference",
            "model: $(ev.model) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(ctx::LogContext, ev::AfterInferenceEvent, idx)
        _log_text!(ctx, "after_inference",
            "model: $(ev.model) | span_id: $(ev.span_id)"; step=idx)
    end

    # BeforeIterationEvent absorbs the start-of-iteration timing bookkeeping
    # that previously lived in a pre-scan loop — we now stash the start time
    # in-line as events stream by, via `ctx.current_time_ns`.
    function log_event(ctx::LogContext, ev::BeforeIterationEvent, _idx)
        _log_text!(ctx, "before_iteration",
            "model: $(ev.model) | iteration: $(ev.iteration) | stop_iteration: $(ev.stop_iteration) | span_id: $(ev.span_id)"; step=ev.iteration)
        ctx.before_times[ev.span_id] = (ev.iteration, ctx.current_time_ns)
    end

    # AfterIterationEvent pairs with the matching BeforeIterationEvent via
    # `span_id` to compute and log the iteration's wall-clock duration.
    function log_event(ctx::LogContext, ev::AfterIterationEvent, _idx)
        _log_text!(ctx, "after_iteration",
            "model: $(ev.model) | iteration: $(ev.iteration) | stop_iteration: $(ev.stop_iteration) | span_id: $(ev.span_id)"; step=ev.iteration)
        if haskey(ctx.before_times, ev.span_id)
            (iter, t0) = ctx.before_times[ev.span_id]
            duration_ms = (ctx.current_time_ns - t0) / 1e6
            ctx.iteration_durations[iter] = duration_ms
            TensorBoardLogger.log_value(ctx.logger, "iteration_time_ms", duration_ms; step=iter)
        end
    end

    function log_event(ctx::LogContext, ev::BeforeDataUpdateEvent, idx)
        _log_text!(ctx, "before_data_update",
            "model: $(ev.model) | data: $(ev.data) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(ctx::LogContext, ev::AfterDataUpdateEvent, idx)
        _log_text!(ctx, "after_data_update",
            "model: $(ev.model) | data: $(ev.data) | span_id: $(ev.span_id)"; step=idx)
    end

    # OnMarginalUpdateEvent carries text, scalar, and distribution logging for
    # the updated marginal. Family-specific behaviour is delegated to the
    # dispatched `_log_posterior_*` helpers above. Scalar and distribution
    # paths use independent try blocks so a failure in one does not suppress
    # the other — and failures surface via `@warn` so they are never silently
    # swallowed (the previous `@debug` hid real errors from the user).
    function log_event(ctx::LogContext, ev::OnMarginalUpdateEvent, idx)
        _log_text!(ctx, "on_marginal_update/$(ev.variable_name)",
            "model: $(ev.model) | variable: $(ev.variable_name) | update: $(ev.update)"; step=idx)
        dist = try
            getdata(ev.update)
        catch err
            @warn "Failed to unwrap marginal" variable_name=ev.variable_name exception=(err, catch_backtrace())
            return nothing
        end
        try
            _log_posterior_scalars!(ctx, dist, ev.variable_name)
        catch err
            @warn "Failed to log posterior scalars" variable_name=ev.variable_name exception=(err, catch_backtrace())
        end
        if ctx.log_distributions
            try
                _log_posterior_distribution!(ctx, dist, ev.variable_name)
            catch err
                @warn "Failed to log posterior distribution" variable_name=ev.variable_name exception=(err, catch_backtrace())
            end
        end
    end

    function log_event(ctx::LogContext, ev::BeforeAutostartEvent, idx)
        _log_text!(ctx, "before_autostart",
            "engine: $(ev.engine) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(ctx::LogContext, ev::AfterAutostartEvent, idx)
        _log_text!(ctx, "after_autostart",
            "engine: $(ev.engine) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(ctx::LogContext, ev::ReactiveMP.BeforeMessageRuleCallEvent, idx)
        _log_text!(ctx, "before_message_rule_call",
            "mapping: $(ev.mapping) | messages: $(ev.messages) | marginals: $(ev.marginals) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(ctx::LogContext, ev::ReactiveMP.AfterMessageRuleCallEvent, idx)
        _log_text!(ctx, "after_message_rule_call",
            "mapping: $(ev.mapping) | messages: $(ev.messages) | marginals: $(ev.marginals) | result: $(ev.result) | annotations: $(ev.annotations) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(ctx::LogContext, ev::ReactiveMP.BeforeProductOfMessagesEvent, idx)
        _log_text!(ctx, "before_product_of_messages",
            "variable: $(ev.variable.label) | context: $(ev.context) | messages: $(ev.messages) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(ctx::LogContext, ev::ReactiveMP.AfterProductOfMessagesEvent, idx)
        _log_text!(ctx, "after_product_of_messages",
            "variable: $(ev.variable.label) | context: $(ev.context) | messages: $(ev.messages) | result: $(ev.result) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(ctx::LogContext, ev::ReactiveMP.BeforeProductOfTwoMessagesEvent, idx)
        _log_text!(ctx, "before_product_of_two_messages",
            "variable: $(ev.variable.label) | context: $(ev.context) | left: $(ev.left) | right: $(ev.right) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(ctx::LogContext, ev::ReactiveMP.AfterProductOfTwoMessagesEvent, idx)
        _log_text!(ctx, "after_product_of_two_messages",
            "variable: $(ev.variable.label) | context: $(ev.context) | left: $(ev.left) | right: $(ev.right) | result: $(ev.result) | annotations: $(ev.annotations) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(ctx::LogContext, ev::ReactiveMP.BeforeMarginalComputationEvent, idx)
        _log_text!(ctx, "before_marginal_computation",
            "variable: $(ev.variable.label) | context: $(ev.context) | messages: $(ev.messages) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(ctx::LogContext, ev::ReactiveMP.AfterMarginalComputationEvent, idx)
        _log_text!(ctx, "after_marginal_computation",
            "variable: $(ev.variable.label) | context: $(ev.context) | messages: $(ev.messages) | result: $(ev.result) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(ctx::LogContext, ev::ReactiveMP.BeforeFormConstraintAppliedEvent, idx)
        _log_text!(ctx, "before_form_constraint_applied",
            "variable: $(ev.variable.label) | context: $(ev.context) | strategy: $(ev.strategy) | distribution: $(ev.distribution) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(ctx::LogContext, ev::ReactiveMP.AfterFormConstraintAppliedEvent, idx)
        _log_text!(ctx, "after_form_constraint_applied",
            "variable: $(ev.variable.label) | context: $(ev.context) | strategy: $(ev.strategy) | distribution: $(ev.distribution) | result: $(ev.result) | span_id: $(ev.span_id)"; step=idx)
    end

    # Fallback for unknown event types
    function log_event(ctx::LogContext, ev::ReactiveMP.Event, idx)
        _log_text!(ctx, "unknown_events",
            "event_type: $(event_name(typeof(ev)))"; step=idx)
    end

    # ─── Main entry point ─────────────────────────────────────────────────────

    """
        convert_to_tensorboard(trace::RxInferTraceCallbacks; output_file::Union{String, Nothing} = nothing,
                               log_distributions::Bool = false, log_text_events::Bool = false,
                               n_samples::Int = 1024)

    Convert trace events from inference to proper TensorFlow event files.

    # Arguments
    - `trace::RxInferTraceCallbacks`: The trace callbacks object from inference results
    - `output_file::Union{String, Nothing}`: Optional directory path to write TensorBoard event logs. If not provided, uses a timestamped directory in the current working directory.
    - `log_distributions::Bool`: When `true`, log each univariate Normal and Gamma posterior as a per-iteration `HistogramSummary` so TensorBoard's **Distributions** tab renders a percentile-band view of the posterior across iterations. The same tag also appears in the **Histograms** tab as an offset ridgeline. Defaults to `false`.
    - `log_text_events::Bool`: When `true`, emit a per-event text breadcrumb (e.g. `before_iteration`, `after_marginal_computation`, and the `Events` step timeline) into the **Text** tab. The `EventCounts` summary is always written regardless of this flag. Scalar and histogram outputs are unaffected. Defaults to `false`.
    - `n_samples::Int`: Number of samples drawn from each posterior to build the per-iteration histogram when `log_distributions=true`. Defaults to 1024.

    # Returns
    - `String`: Path to the directory containing the TensorBoard event log files

    # Description
    This function processes all traced events and creates proper TensorFlow event files using TensorBoardLogger, which can be directly imported and visualized in TensorBoard. Outputs include:
    - Text summaries with event type information and counts
    - Scalar time-series for univariate Normal (`mean`, `precision`) and Gamma (`shape`, `rate`) posteriors
    - Scalar time-series for per-iteration wall-clock duration (`iteration_time_ms`)
    - When `log_distributions=true`: per-iteration `HistogramSummary` under `posteriors/<var>/distribution`, rendered primarily in TensorBoard's Distributions tab

    The output directory can be directly opened in TensorBoard's web interface for visualization and analysis.

    # Example
    ```julia
    results = infer(
        model = my_model(),
        data = my_data,
        trace = true
    )

    trace = results.model.metadata[:trace]

    # Create TensorBoard logs (uses timestamped directory)
    log_dir = convert_to_tensorboard(trace; log_distributions = true)

    # Then run: tensorboard --logdir=\$log_dir
    ```
    """
    function RxInfer.convert_to_tensorboard(trace::RxInferTraceCallbacks;
                                            output_file::Union{String, Nothing} = nothing,
                                            log_distributions::Bool = false,
                                            log_text_events::Bool = false,
                                            n_samples::Int = 1024)

        if isnothing(output_file)
            timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
            output_file = joinpath(pwd(), "tensorboard_logs", timestamp)
        end

        mkpath(output_file)

        events = RxInfer.tracedevents(trace)

        if isempty(events)
            @warn "No events recorded in trace"
            return nothing
        end

        @info "Collected $(length(events)) events from trace"

        logger = TBLogger(output_file)
        ctx    = LogContext(logger;
                            log_distributions=log_distributions,
                            log_text_events=log_text_events,
                            n_samples=n_samples)

        for (idx, traced) in enumerate(events)
            ev     = traced.event
            ev_sym = event_name(typeof(ev))
            ctx.counts[ev_sym]   = get(ctx.counts, ev_sym, 0) + 1
            ctx.current_time_ns  = traced.time_ns
            _log_text!(ctx, "Events", "Step $idx: $(ev_sym)"; step=idx)
            log_event(ctx, ev, idx)
        end

        sorted_counts = sort(collect(ctx.counts), by=first)
        counts_table = reshape(
            vcat(["$(k): $(v)" for (k, v) in sorted_counts], ["total: $(sum(values(ctx.counts)))"]),
            :, 1
        )
        TensorBoardLogger.log_text(ctx.logger, "EventCounts", counts_table; step=1)

        close(logger)

        @info "TensorBoard logs exported to: $output_file"
        @info "Total events logged: $(length(events))"
        @info ""
        @info "To view in TensorBoard, run:"
        @info "  tensorboard --logdir=\"$output_file\""

        return output_file
    end

end
