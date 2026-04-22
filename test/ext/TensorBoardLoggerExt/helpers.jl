using TensorBoardLogger: TensorBoardLogger

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
    log_dir = mktempdir(; cleanup=false)
    try
        fn(log_dir)
    finally
        for attempt in 1:40
            try
                GC.gc(); GC.gc()
                rm(log_dir; recursive=true, force=true)
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
