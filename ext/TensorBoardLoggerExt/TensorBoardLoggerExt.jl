
module TensorBoardLoggerExt
    using RxInfer
    using Dates
    using ReactiveMP: event_name, getdata
    using ExponentialFamily: UnivariateNormalDistributionsFamily, GammaDistributionsFamily
    using Distributions: shape, rate
    using Random: MersenneTwister
    using Statistics: mean, var
    using TensorBoardLogger

    # ─── Distribution helpers (for log_distributions=true) ──────────────────
    # Per-variable range used to fix bin edges across iterations. Uses mean ± 4σ,
    # clamped to the nonnegative reals for Gamma. `nothing` means the distribution
    # is not a supported univariate family and should be skipped.
    function _posterior_range(d)
        if d isa UnivariateNormalDistributionsFamily || d isa GammaDistributionsFamily
            μ = mean(d); σ = sqrt(var(d))
            lo = μ - 4σ; hi = μ + 4σ
            if d isa GammaDistributionsFamily
                lo = max(0.0, lo)
            end
            return (lo, hi)
        end
        return (nothing, nothing)
    end

    # Draw deterministic samples from `dist` to populate the HistogramSummary that
    # drives both the Distributions and Histograms TB dashboards. A seeded
    # MersenneTwister keeps the visualisation reproducible across re-runs.
    function _posterior_samples(dist, n::Int)
        if dist isa UnivariateNormalDistributionsFamily || dist isa GammaDistributionsFamily
            return rand(MersenneTwister(1), dist, n)
        end
        return Float64[]
    end

    # ─── Per-event-type logging methods ───────────────────────────────────────

    function log_event(logger, ev::BeforeModelCreationEvent, idx)
        TensorBoardLogger.log_text(logger, "before_model_creation",
            "span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(logger, ev::AfterModelCreationEvent, idx)
        TensorBoardLogger.log_text(logger, "after_model_creation",
            "model: $(ev.model) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(logger, ev::BeforeInferenceEvent, idx)
        TensorBoardLogger.log_text(logger, "before_inference",
            "model: $(ev.model) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(logger, ev::AfterInferenceEvent, idx)
        TensorBoardLogger.log_text(logger, "after_inference",
            "model: $(ev.model) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(logger, ev::BeforeIterationEvent, _idx)
        TensorBoardLogger.log_text(logger, "before_iteration",
            "model: $(ev.model) | iteration: $(ev.iteration) | stop_iteration: $(ev.stop_iteration) | span_id: $(ev.span_id)"; step=ev.iteration)
    end

    function log_event(logger, ev::AfterIterationEvent, _idx)
        TensorBoardLogger.log_text(logger, "after_iteration",
            "model: $(ev.model) | iteration: $(ev.iteration) | stop_iteration: $(ev.stop_iteration) | span_id: $(ev.span_id)"; step=ev.iteration)
    end

    function log_event(logger, ev::BeforeDataUpdateEvent, idx)
        TensorBoardLogger.log_text(logger, "before_data_update",
            "model: $(ev.model) | data: $(ev.data) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(logger, ev::AfterDataUpdateEvent, idx)
        TensorBoardLogger.log_text(logger, "after_data_update",
            "model: $(ev.model) | data: $(ev.data) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(logger, ev::OnMarginalUpdateEvent, idx)
        TensorBoardLogger.log_text(logger, "on_marginal_update/$(ev.variable_name)",
            "model: $(ev.model) | variable: $(ev.variable_name) | update: $(ev.update)"; step=idx)
    end

    function log_event(logger, ev::BeforeAutostartEvent, idx)
        TensorBoardLogger.log_text(logger, "before_autostart",
            "engine: $(ev.engine) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(logger, ev::AfterAutostartEvent, idx)
        TensorBoardLogger.log_text(logger, "after_autostart",
            "engine: $(ev.engine) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(logger, ev::ReactiveMP.BeforeMessageRuleCallEvent, idx)
        TensorBoardLogger.log_text(logger, "before_message_rule_call",
            "mapping: $(ev.mapping) | messages: $(ev.messages) | marginals: $(ev.marginals) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(logger, ev::ReactiveMP.AfterMessageRuleCallEvent, idx)
        TensorBoardLogger.log_text(logger, "after_message_rule_call",
            "mapping: $(ev.mapping) | messages: $(ev.messages) | marginals: $(ev.marginals) | result: $(ev.result) | annotations: $(ev.annotations) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(logger, ev::ReactiveMP.BeforeProductOfMessagesEvent, idx)
        TensorBoardLogger.log_text(logger, "before_product_of_messages",
            "variable: $(ev.variable.label) | context: $(ev.context) | messages: $(ev.messages) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(logger, ev::ReactiveMP.AfterProductOfMessagesEvent, idx)
        TensorBoardLogger.log_text(logger, "after_product_of_messages",
            "variable: $(ev.variable.label) | context: $(ev.context) | messages: $(ev.messages) | result: $(ev.result) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(logger, ev::ReactiveMP.BeforeProductOfTwoMessagesEvent, idx)
        TensorBoardLogger.log_text(logger, "before_product_of_two_messages",
            "variable: $(ev.variable.label) | context: $(ev.context) | left: $(ev.left) | right: $(ev.right) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(logger, ev::ReactiveMP.AfterProductOfTwoMessagesEvent, idx)
        TensorBoardLogger.log_text(logger, "after_product_of_two_messages",
            "variable: $(ev.variable.label) | context: $(ev.context) | left: $(ev.left) | right: $(ev.right) | result: $(ev.result) | annotations: $(ev.annotations) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(logger, ev::ReactiveMP.BeforeMarginalComputationEvent, idx)
        TensorBoardLogger.log_text(logger, "before_marginal_computation",
            "variable: $(ev.variable.label) | context: $(ev.context) | messages: $(ev.messages) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(logger, ev::ReactiveMP.AfterMarginalComputationEvent, idx)
        TensorBoardLogger.log_text(logger, "after_marginal_computation",
            "variable: $(ev.variable.label) | context: $(ev.context) | messages: $(ev.messages) | result: $(ev.result) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(logger, ev::ReactiveMP.BeforeFormConstraintAppliedEvent, idx)
        TensorBoardLogger.log_text(logger, "before_form_constraint_applied",
            "variable: $(ev.variable.label) | context: $(ev.context) | strategy: $(ev.strategy) | distribution: $(ev.distribution) | span_id: $(ev.span_id)"; step=idx)
    end

    function log_event(logger, ev::ReactiveMP.AfterFormConstraintAppliedEvent, idx)
        TensorBoardLogger.log_text(logger, "after_form_constraint_applied",
            "variable: $(ev.variable.label) | context: $(ev.context) | strategy: $(ev.strategy) | distribution: $(ev.distribution) | result: $(ev.result) | span_id: $(ev.span_id)"; step=idx)
    end

    # Fallback for unknown event types
    function log_event(logger, ev::ReactiveMP.Event, idx)
        TensorBoardLogger.log_text(logger, "unknown_events",
            "event_type: $(event_name(typeof(ev)))"; step=idx)
    end

    # ─── Main entry point ─────────────────────────────────────────────────────

    """
        convert_to_tensorboard(trace::RxInferTraceCallbacks; output_file::Union{String, Nothing} = nothing,
                               log_distributions::Bool = false, n_bins::Int = 64)

    Convert trace events from inference to proper TensorFlow event files.

    # Arguments
    - `trace::RxInferTraceCallbacks`: The trace callbacks object from inference results
    - `output_file::Union{String, Nothing}`: Optional directory path to write TensorBoard event logs. If not provided, uses a timestamped directory in the current working directory.
    - `log_distributions::Bool`: When `true`, log each univariate Normal and Gamma posterior as a per-iteration `HistogramSummary` so TensorBoard's **Distributions** tab renders a percentile-band view of the posterior across iterations. The same tag also appears in the **Histograms** tab as an offset ridgeline. Defaults to `false`.
    - `n_bins::Int`: Number of bins used when `log_distributions=true`. Defaults to 64.

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
                                            n_bins::Int = 64)

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

        # Pre-compute iteration durations from matched before/after pairs via span_id
        iteration_durations = Dict{Int, Float64}()
        before_times = Dict{Any, Tuple{Int, UInt64}}()
        for (idx, traced_event) in enumerate(events)
            ev = traced_event.event
            et = event_name(typeof(ev))
            if et === :before_iteration
                before_times[ev.span_id] = (ev.iteration, traced_event.time_ns)
                #println("step $idx | iteration: $(ev.iteration) | time_ns: $(traced_event.time_ns)")
            elseif et === :after_iteration
                if haskey(before_times, ev.span_id)
                    (iter, t0) = before_times[ev.span_id]
                    iteration_durations[iter] = (traced_event.time_ns - t0) / 1e6
                    #println("step $idx | iteration: $iter | time_ns: $(traced_event.time_ns) | duration_ms: $(iteration_durations[iter])")
                end
            end
        end

        # Pre-scan for stable per-variable bin edges (only when distribution logging is enabled).
        # Stable edges keep bin alignment consistent across iterations so TB's Distributions view
        # shows the posterior sharpening rather than bins drifting.
        posterior_bin_edges = Dict{Symbol, Vector{Float64}}()
        if log_distributions
            ranges = Dict{Symbol, Tuple{Float64, Float64}}()
            for traced_event in events
                ev = traced_event.event
                ev isa OnMarginalUpdateEvent || continue
                try
                    dist = getdata(ev.update)
                    (lo, hi) = _posterior_range(dist)
                    isnothing(lo) && continue
                    prev = get(ranges, ev.variable_name, nothing)
                    ranges[ev.variable_name] = isnothing(prev) ? (lo, hi) : (min(prev[1], lo), max(prev[2], hi))
                catch err
                    @debug "Failed to compute posterior range" variable_name=ev.variable_name exception=err
                end
            end
            for (name, (lo, hi)) in ranges
                if !(hi > lo)
                    eps_width = max(abs(lo), 1.0) * 1e-6
                    lo, hi = lo - eps_width, hi + eps_width
                end
                posterior_bin_edges[name] = collect(range(lo, hi; length = n_bins + 1))
            end
        end

        counts = Dict{Symbol, Int}()
        posterior_step = Dict{Symbol, Int}()

        for (idx, traced_event) in enumerate(events)
            ev = traced_event.event
            event_type = event_name(typeof(ev))
            counts[event_type] = get(counts, event_type, 0) + 1

            TensorBoardLogger.log_text(logger, "Events", "Step $idx: $(event_type)"; step=idx)
            log_event(logger, ev, idx)

            # Log iteration wall-clock time as a scalar
            if ev isa AfterIterationEvent && haskey(iteration_durations, ev.iteration)
                TensorBoardLogger.log_value(logger, "iteration_time_ms", iteration_durations[ev.iteration]; step=ev.iteration)
            end

            # Log univariate Normal (mean, precision) and Gamma (shape, rate) posteriors as scalars
            if ev isa OnMarginalUpdateEvent
                try
                    dist = getdata(ev.update)
                    if dist isa UnivariateNormalDistributionsFamily
                        posterior_step[ev.variable_name] = get(posterior_step, ev.variable_name, 0) + 1
                        step = posterior_step[ev.variable_name]
                        TensorBoardLogger.log_value(logger, "posteriors/$(ev.variable_name)/mean", mean(dist); step=step)
                        TensorBoardLogger.log_value(logger, "posteriors/$(ev.variable_name)/precision", inv(var(dist)); step=step)
                    elseif dist isa GammaDistributionsFamily
                        posterior_step[ev.variable_name] = get(posterior_step, ev.variable_name, 0) + 1
                        step = posterior_step[ev.variable_name]
                        TensorBoardLogger.log_value(logger, "posteriors/$(ev.variable_name)/shape", shape(dist); step=step)
                        TensorBoardLogger.log_value(logger, "posteriors/$(ev.variable_name)/rate",  rate(dist);  step=step)
                    end
                catch err
                    @debug "Failed to log posterior scalars" variable_name=ev.variable_name exception=err
                end

                # Log per-iteration HistogramSummary so TB's Distributions tab renders a
                # percentile-band view of each posterior across iterations.
                if log_distributions && haskey(posterior_bin_edges, ev.variable_name)
                    try
                        dist  = getdata(ev.update)
                        step  = get(posterior_step, ev.variable_name, 0)
                        edges = posterior_bin_edges[ev.variable_name]
                        n_samples = max(512, 8 * n_bins)
                        samples = _posterior_samples(dist, n_samples)
                        if !isempty(samples)
                            TensorBoardLogger.log_histogram(logger, "posteriors/$(ev.variable_name)/distribution",
                                (edges, samples); step = step)
                        end
                    catch err
                        @debug "Failed to log posterior distribution" variable_name=ev.variable_name exception=err
                    end
                end
            end
        end

        sorted_counts = sort(collect(counts), by=first)
        counts_table = reshape(
            vcat(["$(k): $(v)" for (k, v) in sorted_counts], ["total: $(sum(values(counts)))"]),
            :, 1
        )
        TensorBoardLogger.log_text(logger, "EventCounts", counts_table; step=1)

        close(logger)

        @info "TensorBoard logs exported to: $output_file"
        @info "Total events logged: $(length(events))"
        @info ""
        @info "To view in TensorBoard, run:"
        @info "  tensorboard --logdir=\"$output_file\""

        return output_file
    end

end
