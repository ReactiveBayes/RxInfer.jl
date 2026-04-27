
module TensorBoardLoggerExt
using RxInfer
using ReactiveMP: event_name, getdata
using ExponentialFamily:
    UnivariateNormalDistributionsFamily, GammaDistributionsFamily
using Distributions:
    UnivariateDistribution,
    Beta,
    Bernoulli,
    Binomial,
    InverseGamma,
    Poisson,
    Geometric,
    NegativeBinomial,
    Exponential,
    VonMises,
    Weibull,
    LogNormal,
    Erlang,
    Laplace,
    Pareto,
    Rayleigh,
    Chisq,
    shape,
    rate,
    scale,
    params,
    succprob,
    ntrials,
    location,
    dof
using Dates: now, format
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
    log_posteriors::Union{Bool, Set{Symbol}}
    log_distributions::Bool
    log_text_events::Bool
    n_samples::Int
    current_time_ns::UInt64
    # ─── Tier 1 timing summary ───────────────────────────────────────────
    # Singleton spans (one model-creation, one inference per `infer` call) so
    # we don't need a span_id-keyed dict — just remember the start `time_ns`
    # and compute the wall-clock delta on the matching `After*` event.
    # `0` means "the matching Before* event has not been seen", `NaN` ms
    # means "duration not measured" — both are encoded as separate fields so
    # the summary writer can distinguish missing vs. zero-duration runs.
    model_build_start_ns::UInt64
    model_build_ms::Float64
    inference_start_ns::UInt64
    inference_ms::Float64
    # First/last `time_ns` across all traced events — drives the run-wide
    # wall-clock figure in the Summary text tag.
    first_event_ns::UInt64
    last_event_ns::UInt64
end

# Normalize the user-facing `log_posteriors` value to the internal
# representation. `Bool` passes through; any vector of names is collapsed
# into a `Set{Symbol}` so per-event filtering is O(1) and accepts either
# `String` (`["μ", "θ"]`) or `Symbol` (`[:μ, :θ]`) inputs interchangeably.
_normalize_posteriors(p::Bool) = p
_normalize_posteriors(v::AbstractVector) = Set{Symbol}(Symbol(x) for x in v)

LogContext(logger; log_posteriors::Union{Bool, AbstractVector{<:Union{Symbol, AbstractString}}} = true, log_distributions::Bool, log_text_events::Bool, n_samples::Int) = LogContext(
    logger,
    Dict{Int, Float64}(),
    Dict{Any, Tuple{Int, UInt64}}(),
    Dict{Symbol, Int}(),
    Dict{Symbol, Int}(),
    _normalize_posteriors(log_posteriors),
    log_distributions,
    log_text_events,
    n_samples,
    zero(UInt64),
    zero(UInt64),
    NaN,
    zero(UInt64),
    NaN,
    zero(UInt64),
    zero(UInt64),
)

# Central gate for all narrative/event text summaries. Scalar and
# histogram logging stays unconditional — only the per-event text
# breadcrumbs are opt-in via `log_text_events`.
@inline _log_text!(ctx::LogContext, tag, msg; step) =
    ctx.log_text_events &&
    TensorBoardLogger.log_text(ctx.logger, tag, msg; step = step)

# Per-variable gate for posterior scalar + histogram logging. Dispatches on
# the runtime type of `log_posteriors`: `Bool` is the global on/off, and
# `Set{Symbol}` restricts logging to an explicit allow-list of variable
# names. Empty set behaves like `false` (logs nothing).
@inline _should_log_posterior(ctx::LogContext, name::Symbol) = _check_posterior(
    ctx.log_posteriors, name
)
@inline _check_posterior(flag::Bool, ::Symbol) = flag
@inline _check_posterior(allowed::Set{Symbol}, n::Symbol) = n in allowed

# Pipe-delimited `"k1: v1 | k2: v2 | ..."` formatter for text-event
# breadcrumbs. Reads each named field via `getfield(ev, f)`, so this helper
# only fits events where every logged value is a direct field access.
# Events that need `ev.variable.label` or a renamed label (e.g.
# `OnMarginalUpdateEvent`'s `variable: $(ev.variable_name)`) build the
# string inline.
@inline _format_fields(ev, fields::NTuple{N, Symbol}) where {N} = join(
    ("$(f): $(getfield(ev, f))" for f in fields), " | "
)

# ─── Distribution-family dispatched helpers ──────────────────────────────
# Deterministic samples via a seeded MersenneTwister keep the HistogramSummary
# reproducible across re-runs. The `::Any` fallback returns an empty vector
# so non-univariate or unsupported marginals are silently skipped without
# any branching at the call site.
_posterior_samples(dist::UnivariateDistribution, n::Int) = rand(MersenneTwister(1), dist, n)
_posterior_samples(::Any, ::Int)                         = Float64[]

# Per-distribution scalar tag table. Returns a `NamedTuple` of
# `(tag => value)` pairs to log under `posteriors/<name>/<tag>`.
# `nothing` means "skip this variable" — distinct from an empty NamedTuple,
# which would still bump the step counter. Family-specific methods take
# precedence over the `UnivariateDistribution` generic fallback (which
# logs `mean` and `var` so any unspecialised univariate marginal still
# shows convergence behaviour in TensorBoard).
_posterior_tags(::Any) = nothing

_posterior_tags(d::UnivariateNormalDistributionsFamily) = (
    mean = mean(d), precision = inv(var(d))
)

_posterior_tags(d::GammaDistributionsFamily) = (
    shape = shape(d), rate = rate(d)
)

function _posterior_tags(d::Beta)
    α, β = params(d)
    return (alpha = α, beta = β, mean = α / (α + β))
end

_posterior_tags(d::Bernoulli) = (succprob = succprob(d),)
_posterior_tags(d::Binomial) = (ntrials = ntrials(d), succprob = succprob(d))
_posterior_tags(d::InverseGamma) = (shape = shape(d), scale = scale(d))
_posterior_tags(d::Poisson) = (rate = rate(d),)
_posterior_tags(d::Geometric) = (succprob = succprob(d),)

function _posterior_tags(d::NegativeBinomial)
    r, _ = params(d)
    return (r = r, succprob = succprob(d))
end

_posterior_tags(d::Exponential) = (rate = rate(d),)

function _posterior_tags(d::VonMises)
    μ, κ = params(d)
    return (location = μ, concentration = κ)
end

_posterior_tags(d::Weibull) = (shape = shape(d), scale = scale(d))

function _posterior_tags(d::LogNormal)
    meanlog, stdlog = params(d)
    return (meanlog = meanlog, stdlog = stdlog)
end

_posterior_tags(d::Erlang) = (shape = shape(d), scale = scale(d))
_posterior_tags(d::Laplace) = (location = location(d), scale = scale(d))
_posterior_tags(d::Pareto) = (shape = shape(d), scale = scale(d))
_posterior_tags(d::Rayleigh) = (scale = scale(d),)
_posterior_tags(d::Chisq) = (dof = dof(d),)

# Generic moment fallback: any UnivariateDistribution we haven't
# special-cased still gets `mean`/`var` tags so TensorBoard shows
# convergence behaviour instead of going silent.
_posterior_tags(d::UnivariateDistribution) = (mean = mean(d), var = var(d))

# Scalar-posterior logging delegates to `_posterior_tags(dist)` and writes
# each `(tag => value)` pair under `posteriors/<name>/<tag>`. The step
# counter is bumped only when there is at least one tag to log, so
# unsupported distributions don't desync the per-variable step.
function _log_posterior_scalars!(ctx::LogContext, dist, name::Symbol)
    tags = _posterior_tags(dist)
    tags === nothing && return nothing
    step = (ctx.posterior_step[name] = get(ctx.posterior_step, name, 0) + 1)
    for (tag, value) in pairs(tags)
        TensorBoardLogger.log_value(
            ctx.logger, "posteriors/$(name)/$(tag)", value; step = step
        )
    end
    return nothing
end

# Per-iteration HistogramSummary. The data-only `log_histogram` overload
# lets TB auto-bin each iteration's samples so HistogramProto.min/max track
# the actual sample extremes — that is what makes the Distributions plugin
# narrow the percentile bands as the posterior sharpens.
function _log_posterior_distribution!(
    ctx::LogContext, dist::UnivariateDistribution, name::Symbol
)
    samples = _posterior_samples(dist, ctx.n_samples)
    isempty(samples) && return nothing
    step = get(ctx.posterior_step, name, 0)
    TensorBoardLogger.log_histogram(
        ctx.logger, "posteriors/$(name)/distribution", samples; step = step
    )
end
_log_posterior_distribution!(::LogContext, ::Any, ::Symbol) = nothing

# ─── Run summary writer ───────────────────────────────────────────────────
# Emits a single text tag (`Summary`) at step 1 with a one-column table of
# `key: value` lines — same render layout as `EventCounts`. Lines that
# correspond to unmeasured fields (no matching Before/After event seen,
# or no iteration durations recorded) are silently skipped so the table
# never advertises misleading zero-duration values for missing data.
@inline _fmt_ms(x::Float64) = string(round(x; digits = 3), " ms")

function _log_summary!(ctx::LogContext)
    lines = String[]
    if !isnan(ctx.model_build_ms)
        push!(lines, "model_build: $(_fmt_ms(ctx.model_build_ms))")
    end
    if !isnan(ctx.inference_ms)
        push!(lines, "inference: $(_fmt_ms(ctx.inference_ms))")
    end
    if ctx.first_event_ns != zero(UInt64) &&
        ctx.last_event_ns >= ctx.first_event_ns
        wall_ms = (ctx.last_event_ns - ctx.first_event_ns) / 1e6
        push!(lines, "total_wall: $(_fmt_ms(wall_ms))")
    end
    if !isempty(ctx.iteration_durations)
        durations = collect(values(ctx.iteration_durations))
        push!(lines, "n_iterations: $(length(durations))")
        push!(lines, "iter_total: $(_fmt_ms(sum(durations)))")
        push!(lines, "iter_mean: $(_fmt_ms(sum(durations) / length(durations)))")
        push!(lines, "iter_min: $(_fmt_ms(minimum(durations)))")
        push!(lines, "iter_max: $(_fmt_ms(maximum(durations)))")
    end
    isempty(lines) && return nothing
    TensorBoardLogger.log_text(
        ctx.logger, "Summary", reshape(lines, :, 1); step = 1
    )
    return nothing
end

# ─── Per-event-type logging methods ───────────────────────────────────────

function log_event(ctx::LogContext, ev::BeforeModelCreationEvent, idx)
    ctx.model_build_start_ns = ctx.current_time_ns
    _log_text!(
        ctx,
        "before_model_creation",
        _format_fields(ev, (:span_id,));
        step = idx,
    )
end

function log_event(ctx::LogContext, ev::AfterModelCreationEvent, idx)
    if ctx.model_build_start_ns != zero(UInt64)
        ctx.model_build_ms =
            (ctx.current_time_ns - ctx.model_build_start_ns) / 1e6
    end
    _log_text!(
        ctx,
        "after_model_creation",
        _format_fields(ev, (:model, :span_id));
        step = idx,
    )
end

function log_event(ctx::LogContext, ev::BeforeInferenceEvent, idx)
    ctx.inference_start_ns = ctx.current_time_ns
    _log_text!(
        ctx,
        "before_inference",
        _format_fields(ev, (:model, :span_id));
        step = idx,
    )
end

function log_event(ctx::LogContext, ev::AfterInferenceEvent, idx)
    if ctx.inference_start_ns != zero(UInt64)
        ctx.inference_ms =
            (ctx.current_time_ns - ctx.inference_start_ns) / 1e6
    end
    _log_text!(
        ctx,
        "after_inference",
        _format_fields(ev, (:model, :span_id));
        step = idx,
    )
end

# BeforeIterationEvent absorbs the start-of-iteration timing bookkeeping
# that previously lived in a pre-scan loop — we now stash the start time
# in-line as events stream by, via `ctx.current_time_ns`.
function log_event(ctx::LogContext, ev::BeforeIterationEvent, _idx)
    _log_text!(
        ctx,
        "before_iteration",
        _format_fields(ev, (:model, :iteration, :stop_iteration, :span_id));
        step = ev.iteration,
    )
    ctx.before_times[ev.span_id] = (ev.iteration, ctx.current_time_ns)
end

# AfterIterationEvent pairs with the matching BeforeIterationEvent via
# `span_id` to compute and log the iteration's wall-clock duration.
function log_event(ctx::LogContext, ev::AfterIterationEvent, _idx)
    _log_text!(
        ctx,
        "after_iteration",
        _format_fields(ev, (:model, :iteration, :stop_iteration, :span_id));
        step = ev.iteration,
    )
    if haskey(ctx.before_times, ev.span_id)
        (iter, t0) = ctx.before_times[ev.span_id]
        duration_ms = (ctx.current_time_ns - t0) / 1e6
        ctx.iteration_durations[iter] = duration_ms
        TensorBoardLogger.log_value(
            ctx.logger, "iteration_time_ms", duration_ms; step = iter
        )
    end
end

function log_event(ctx::LogContext, ev::BeforeDataUpdateEvent, idx)
    _log_text!(
        ctx,
        "before_data_update",
        _format_fields(ev, (:model, :data, :span_id));
        step = idx,
    )
end

function log_event(ctx::LogContext, ev::AfterDataUpdateEvent, idx)
    _log_text!(
        ctx,
        "after_data_update",
        _format_fields(ev, (:model, :data, :span_id));
        step = idx,
    )
end

# OnMarginalUpdateEvent carries text, scalar, and distribution logging for
# the updated marginal. Family-specific behaviour is delegated to the
# dispatched `_log_posterior_*` helpers above. Scalar and distribution
# paths use independent try blocks so a failure in one does not suppress
# the other — and failures surface via `@warn` so they are never silently
# swallowed (the previous `@debug` hid real errors from the user).
function log_event(ctx::LogContext, ev::OnMarginalUpdateEvent, idx)
    _log_text!(
        ctx,
        "on_marginal_update/$(ev.variable_name)",
        "model: $(ev.model) | variable: $(ev.variable_name) | update: $(ev.update)";
        step = idx,
    )
    _should_log_posterior(ctx, ev.variable_name) || return nothing
    dist = try
        getdata(ev.update)
    catch err
        @warn "Failed to unwrap marginal" variable_name = ev.variable_name exception = (
            err, catch_backtrace()
        )
        return nothing
    end
    try
        _log_posterior_scalars!(ctx, dist, ev.variable_name)
    catch err
        @warn "Failed to log posterior scalars" variable_name = ev.variable_name exception = (
            err, catch_backtrace()
        )
    end
    if ctx.log_distributions
        try
            _log_posterior_distribution!(ctx, dist, ev.variable_name)
        catch err
            @warn "Failed to log posterior distribution" variable_name =
                ev.variable_name exception = (err, catch_backtrace())
        end
    end
end

function log_event(ctx::LogContext, ev::BeforeAutostartEvent, idx)
    _log_text!(
        ctx,
        "before_autostart",
        _format_fields(ev, (:engine, :span_id));
        step = idx,
    )
end

function log_event(ctx::LogContext, ev::AfterAutostartEvent, idx)
    _log_text!(
        ctx,
        "after_autostart",
        _format_fields(ev, (:engine, :span_id));
        step = idx,
    )
end

function log_event(
    ctx::LogContext, ev::ReactiveMP.BeforeMessageRuleCallEvent, idx
)
    _log_text!(
        ctx,
        "before_message_rule_call",
        _format_fields(ev, (:mapping, :messages, :marginals, :span_id));
        step = idx,
    )
end

function log_event(
    ctx::LogContext, ev::ReactiveMP.AfterMessageRuleCallEvent, idx
)
    _log_text!(
        ctx,
        "after_message_rule_call",
        _format_fields(
            ev,
            (:mapping, :messages, :marginals, :result, :annotations, :span_id),
        );
        step = idx,
    )
end

function log_event(
    ctx::LogContext, ev::ReactiveMP.BeforeProductOfMessagesEvent, idx
)
    _log_text!(
        ctx,
        "before_product_of_messages",
        "variable: $(ev.variable.label) | context: $(ev.context) | messages: $(ev.messages) | span_id: $(ev.span_id)";
        step = idx,
    )
end

function log_event(
    ctx::LogContext, ev::ReactiveMP.AfterProductOfMessagesEvent, idx
)
    _log_text!(
        ctx,
        "after_product_of_messages",
        "variable: $(ev.variable.label) | context: $(ev.context) | messages: $(ev.messages) | result: $(ev.result) | span_id: $(ev.span_id)";
        step = idx,
    )
end

function log_event(
    ctx::LogContext, ev::ReactiveMP.BeforeProductOfTwoMessagesEvent, idx
)
    _log_text!(
        ctx,
        "before_product_of_two_messages",
        "variable: $(ev.variable.label) | context: $(ev.context) | left: $(ev.left) | right: $(ev.right) | span_id: $(ev.span_id)";
        step = idx,
    )
end

function log_event(
    ctx::LogContext, ev::ReactiveMP.AfterProductOfTwoMessagesEvent, idx
)
    _log_text!(
        ctx,
        "after_product_of_two_messages",
        "variable: $(ev.variable.label) | context: $(ev.context) | left: $(ev.left) | right: $(ev.right) | result: $(ev.result) | annotations: $(ev.annotations) | span_id: $(ev.span_id)";
        step = idx,
    )
end

function log_event(
    ctx::LogContext, ev::ReactiveMP.BeforeMarginalComputationEvent, idx
)
    _log_text!(
        ctx,
        "before_marginal_computation",
        "variable: $(ev.variable.label) | context: $(ev.context) | messages: $(ev.messages) | span_id: $(ev.span_id)";
        step = idx,
    )
end

function log_event(
    ctx::LogContext, ev::ReactiveMP.AfterMarginalComputationEvent, idx
)
    _log_text!(
        ctx,
        "after_marginal_computation",
        "variable: $(ev.variable.label) | context: $(ev.context) | messages: $(ev.messages) | result: $(ev.result) | span_id: $(ev.span_id)";
        step = idx,
    )
end

function log_event(
    ctx::LogContext, ev::ReactiveMP.BeforeFormConstraintAppliedEvent, idx
)
    _log_text!(
        ctx,
        "before_form_constraint_applied",
        "variable: $(ev.variable.label) | context: $(ev.context) | strategy: $(ev.strategy) | distribution: $(ev.distribution) | span_id: $(ev.span_id)";
        step = idx,
    )
end

function log_event(
    ctx::LogContext, ev::ReactiveMP.AfterFormConstraintAppliedEvent, idx
)
    _log_text!(
        ctx,
        "after_form_constraint_applied",
        "variable: $(ev.variable.label) | context: $(ev.context) | strategy: $(ev.strategy) | distribution: $(ev.distribution) | result: $(ev.result) | span_id: $(ev.span_id)";
        step = idx,
    )
end

# Fallback for unknown event types
function log_event(ctx::LogContext, ev::ReactiveMP.Event, idx)
    _log_text!(
        ctx,
        "unknown_events",
        "event_type: $(event_name(typeof(ev)))";
        step = idx,
    )
end

# defined in RxInfer.jl in src/callbacks/trace.jl
# this module extends it
function RxInfer.convert_to_tensorboard(
    trace::RxInferTraceCallbacks;
    output_file::Union{String, Nothing} = nothing,
    log_posteriors::Union{
        Bool, AbstractVector{<:Union{Symbol, AbstractString}}
    } = true,
    log_distributions::Bool = false,
    log_text_events::Bool = false,
    n_samples::Int = 1024,
    verbose = true,
)
    if isnothing(output_file)
        output_file = joinpath(pwd(), "tensorboard_logs")
    end

    mkpath(output_file)

    events = RxInfer.tracedevents(trace)

    if isempty(events)
        @warn "No events recorded in trace"
        return nothing
    end

    if verbose
        @info "Collected $(length(events)) events from trace"
    end

    log_subdir = joinpath(output_file, format(now(), "yyyy-mm-dd_HH-MM-SS"))
    mkpath(log_subdir)
    logger = TBLogger(log_subdir, tb_append)
    ctx    = LogContext(logger; log_posteriors = log_posteriors, log_distributions = log_distributions, log_text_events = log_text_events, n_samples = n_samples)

    for (idx, traced) in enumerate(events)
        ev                  = traced.event
        ev_sym              = event_name(typeof(ev))
        ctx.counts[ev_sym]  = get(ctx.counts, ev_sym, 0) + 1
        ctx.current_time_ns = traced.time_ns
        if ctx.first_event_ns == zero(UInt64)
            ctx.first_event_ns = traced.time_ns
        end
        ctx.last_event_ns = traced.time_ns
        _log_text!(ctx, "Events", "Step $idx: $(ev_sym)"; step = idx)
        log_event(ctx, ev, idx)
    end

    sorted_counts = sort(collect(ctx.counts); by = first)
    counts_table = reshape(
        vcat(
            ["$(k): $(v)" for (k, v) in sorted_counts],
            ["total: $(sum(values(ctx.counts)))"],
        ),
        :,
        1,
    )
    TensorBoardLogger.log_text(
        ctx.logger, "EventCounts", counts_table; step = 1
    )

    _log_summary!(ctx)

    close(logger)
    # Drop internal references to the closed IOStreams so any lingering
    # Windows file-lock isn't held past this function's return. `mktempdir`
    # cleanup in tests races `rm` against the OS releasing the handle, and
    # emits an @error that VSCode's test-item runner surfaces as red.
    empty!(logger.all_files)
    GC.gc()

    if verbose
        @info "TensorBoard logs exported to: $log_subdir"
        @info "Total events logged: $(length(events))"
        @info ""
        @info "To view in TensorBoard, run:"
        @info "  tensorboard --logdir=\"$output_file\""
    end

    return log_subdir
end

end
