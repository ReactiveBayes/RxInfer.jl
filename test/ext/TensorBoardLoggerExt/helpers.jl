using TensorBoardLogger: TensorBoardLogger, TBLogger, tb_append

# ─── Tempdir cleanup retry ────────────────────────────────────────────────
# Windows file-handle release is asynchronous: TensorBoardLogger readers
# and the ProtoBuf decoder keep the `.tfevents` file mapped for some time
# after `close`. Julia's `mktempdir(fn)` tries to `rm` the directory in a
# `finally` block and emits a loud `@error` when the unlink fails with
# EBUSY — the VSCode test-item runner surfaces that error as a red failure
# even though every `@test` passed.
#
# `with_safe_tempdir` runs with `cleanup=false`, then retries `rm` until
# the OS releases the handle. If cleanup still fails after the retry
# budget we leave the directory behind (the OS temp sweeper will reclaim
# it) rather than failing the test.
function with_safe_tempdir(fn)
    log_dir = mktempdir(; cleanup = false)
    try
        fn(log_dir)
    finally
        for attempt in 1:40
            try
                GC.gc();
                GC.gc()
                rm(log_dir; recursive = true, force = true)
                break
            catch
                attempt == 40 || sleep(0.05)
            end
        end
    end
end

# ─── Event-file bypass readers ────────────────────────────────────────────
# TensorBoardLogger 0.1.26's `deserialize_tensor_summary` (used internally
# by `tags()` and `map_summaries()`) still reads `summary.tensor` — a
# field that no longer exists on the regenerated `Summary.Value`, where
# the tensor now lives at `summary.value.value` inside a `OneOf`. Any
# logdir containing a text summary (we always write `EventCounts`) throws
# `FieldError` when iterated through that path.
#
# The tests only need tag *names* and *step* numbers, not decoded tensor
# payloads. These helpers iterate event files directly and access the
# plain `summary.tag` / `event.step` fields, skipping the broken
# deserializer entirely.
function _each_summary(fn, logdir)
    for event_file in TensorBoardLogger.TBEventFileCollectionIterator(logdir)
        for event in event_file
            event.what === nothing && continue
            s = event.what.value
            isa(s, TensorBoardLogger.Summary) || continue
            for summary_value in s.value
                fn(event.step, summary_value.tag)
            end
        end
    end
end

function read_tags(logdir)
    tags = Set{String}()
    _each_summary(logdir) do _, tag
        push!(tags, tag)
    end
    return tags
end

function steps_for_tag(logdir, tag)
    steps = BitSet()
    _each_summary(logdir) do step, t
        t == tag && push!(steps, step)
    end
    return steps
end

# ─── Direct-dispatch logger fixture ───────────────────────────────────────
# Every tier-2/3 distribution test shares the same ritual: open a TBLogger
# in a safe tempdir, build a LogContext with text/distribution logging
# off, push events through `_log_posterior_scalars!`, then close and
# release Windows file handles before reading tags. Inlining that into
# every testitem produced ~14× the same 15-line skeleton.
#
# `with_dispatch_logger` runs the user's write function, flushes the
# logger, and snapshots the tag set + per-tag step set BEFORE the tempdir
# teardown. Returning a NamedTuple of `(tags, steps)` keeps the call site
# linear: write events in the do-block, then assert against the result.
function with_dispatch_logger(write_events; n_samples::Int = 0)
    ext = Base.get_extension(RxInfer, :TensorBoardLoggerExt)
    ext === nothing && error("TensorBoardLoggerExt not loaded")

    tags = Set{String}()
    steps = Dict{String, BitSet}()

    with_safe_tempdir() do log_dir
        logger = TBLogger(log_dir, tb_append)
        ctx = ext.LogContext(
            logger;
            log_distributions = false,
            log_text_events   = false,
            n_samples         = n_samples,
        )
        try
            write_events(ext, ctx)
        finally
            close(logger)
            empty!(logger.all_files)
            GC.gc()
        end
        # Snapshot before `with_safe_tempdir` tears the directory down.
        tags = read_tags(log_dir)
        for tag in tags
            steps[tag] = steps_for_tag(log_dir, tag)
        end
    end

    return (tags = tags, steps = steps)
end

# ─── Shared model fixtures ────────────────────────────────────────────────
# Three reference models drive every end-to-end test in this directory.
# Each `*_inference` factory returns a real `infer(...; trace = true)`
# result so callers can read `results.model.metadata[:trace]` and feed it
# to `convert_to_tensorboard`. Iteration count and dataset size are
# tunable; defaults match the most common values used historically.
#
# Splatting `kwargs...` into `infer` lets callers pass through extras
# (e.g. `returnvars = (θ = KeepEach(),)` for the Beta-Binomial workflow).

@model function iid_normal_model(y)
    μ ~ Normal(; mean = 0.0, precision = 0.1)
    τ ~ Gamma(; shape = 1.0, rate = 1.0)
    y .~ Normal(; mean = μ, precision = τ)
end

@constraints function iid_normal_constraints()
    q(μ, τ) = q(μ)q(τ)
end

function iid_normal_inference(; iterations::Int = 2, n_samples::Int = 10, seed::Int = 42, kwargs...)
    initialization = @initialization begin
        q(μ) = vague(NormalMeanPrecision)
        q(τ) = vague(GammaShapeRate)
    end
    dataset = rand(StableRNG(seed), NormalMeanPrecision(3.1415, 2.7182), n_samples)
    return infer(;
        model          = iid_normal_model(),
        data           = (y = dataset,),
        constraints    = iid_normal_constraints(),
        iterations     = iterations,
        initialization = initialization,
        trace          = true,
        kwargs...,
    )
end

@model function coin_toss_model(y)
    θ ~ Beta(1.0, 1.0)
    y .~ Bernoulli(θ)
end

function coin_toss_inference(; iterations::Int = 2, n_samples::Int = 10, seed::Int = 42, kwargs...)
    initialization = @initialization begin
        q(θ) = vague(Beta)
    end
    dataset = rand(StableRNG(seed), Bernoulli(0.7), n_samples)
    return infer(;
        model          = coin_toss_model(),
        data           = (y = dataset,),
        iterations     = iterations,
        initialization = initialization,
        trace          = true,
        kwargs...,
    )
end

@model function iid_invgamma_model(y)
    μ ~ Normal(mean = 0.0, variance = 100.0)
    σ² ~ GammaInverse(α = 2.0, θ = 1.0)
    y .~ Normal(mean = μ, variance = σ²)
end

@constraints function iid_invgamma_constraints()
    q(μ, σ²) = q(μ)q(σ²)
end

function iid_invgamma_inference(; iterations::Int = 2, n_samples::Int = 10, seed::Int = 42, kwargs...)
    initialization = @initialization begin
        q(μ) = vague(NormalMeanVariance)
        q(σ²) = vague(GammaInverse)
    end
    dataset = rand(StableRNG(seed), NormalMeanVariance(3.1415, 1.0 / 2.7182), n_samples)
    return infer(;
        model          = iid_invgamma_model(),
        data           = (y = dataset,),
        constraints    = iid_invgamma_constraints(),
        iterations     = iterations,
        initialization = initialization,
        trace          = true,
        kwargs...,
    )
end
