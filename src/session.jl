using Dates, UUIDs, Preferences

mutable struct SessionInvoke
    id::UUID
    label::Symbol
    status::Symbol
    execution_start::DateTime
    execution_end::DateTime
    context::Dict{Symbol, Any}
end

"""
    Session

A structure that maintains a log of all inference invocations during a RxInfer session.
Each session has a unique identifier and tracks when it was created. The session stores
a history of all session invocations (`SessionInvoke`) that occurred during its lifetime.

# Fields
- `id::UUID`: A unique identifier for the session
- `created_at::DateTime`: Timestamp when the session was created
- `environment::Dict{Symbol, Any}`: Information about the Julia & RxInfer versions and system when the session was created
- `invokes::Vector{SessionInvoke}`: List of all inference invocations that occurred during the session

The session logging is transparent and only collects non-sensitive information about calls.
Users can inspect the session at any time using `get_current_session()` and reset it using `reset_session!()`.
"""
struct Session
    id::UUID
    created_at::DateTime
    environment::Dict{Symbol, Any}
    invokes::Vector{SessionInvoke}
end

"""
    create_session()

Create a new session with a unique identifier and current timestamp.

# Returns
- `Session`: A new session instance with no inference invocations recorded
"""
function create_session()
    environment = Dict{Symbol, Any}(
        :julia_version => string(VERSION),
        :rxinfer_version => string(pkgversion(RxInfer)),
        :os => string(Sys.KERNEL),
        :machine => string(Sys.MACHINE),
        :cpu_threads => Sys.CPU_THREADS,
        :word_size => Sys.WORD_SIZE
    )
    return Session(
        uuid4(),          # Generate unique ID
        now(),            # Current timestamp
        environment,      # Environment information
        SessionInvoke[]     # Empty vector of invokes
    )
end

"""
    create_invoke(label::Symbol)

Create a new session invoke with the given label.
"""
function create_invoke(label::Symbol)
    return SessionInvoke(uuid4(), label, :unknown, Dates.now(), Dates.now(), Dict{Symbol, Any}())
end

"""
    with_session(f::F, session, label::Symbol = :unknown) where {F}

Execute function `f` within a session context with the specified label. If `session` is provided, logs execution details including timing and errors.
If `session` is `nothing`, executes `f` without logging.
"""
function with_session(f::F, session, label::Symbol = :unknown) where {F}
    if isnothing(session)
        return f(nothing)
    elseif session isa Session
        invoke = create_invoke(label)
        try
            result = f(invoke)
            invoke.status = :success
            invoke.execution_end = Dates.now()
            push!(session.invokes, invoke)
            return result
        catch e
            invoke.status = :failed
            invoke.context[:error] = string(e)
            push!(session.invokes, invoke)
            rethrow(e)
        end
    else
        error(lazy"Unsupported session type $(typeof(session)). Should either be `RxInfer.Session` or `nothing`.")
    end
end

"""
    append_invoke_context(f, invoke)

Append context information to a session invoke. If `invoke` is a `SessionInvoke`, executes function `f` with the invoke's context.
If `invoke` is `nothing`, does nothing.
"""
function append_invoke_context end

append_invoke_context(f::F, ::Nothing) where {F} = nothing
append_invoke_context(f::F, invoke::SessionInvoke) where {F} = f(invoke.context)

const default_session_sem = Base.Semaphore(1)
# The `Ref` is initialized in the __init__ function based on user preferences
const default_session_ref = Ref{Union{Nothing, Session}}(nothing)

"""
    default_session()::Union{Nothing, Session}

Get the current default session. If no session exists, returns `nothing`.

# Returns
- `Union{Nothing, Session}`: The current default session or `nothing` if logging is disabled
"""
function default_session()::Union{Nothing, Session}
    return Base.acquire(default_session_sem) do
        return default_session_ref[]
    end
end

"""
    set_default_session!(session::Union{Nothing, Session})

Set the default session to a new session or disable logging by passing `nothing`. 

# Arguments
- `session::Union{Nothing, Session}`: The new session to set as default, or `nothing` to disable logging
"""
function set_default_session!(session::Union{Nothing, Session})
    return Base.acquire(default_session_sem) do
        default_session_ref[] = session
        return session
    end
end

"""
Disables session logging for RxInfer globally at compile time and saves it in package preferences. Has effect after Julia restart.

Restart Julia and verify it by `isnothing(RxInfer.default_session())`. 

Note that session logging still can be enabled manually for the current session if `set_default_session!` is called manually with appropriate `Session` object. 
"""
function disable_session_logging!()
    @set_preferences!("enable_session_logging" => false)
    @info "Disabled session logging. Changes will take effect after Julia restart."
end

"""
Enables session logging for RxInfer globally at compile time and saves it in package preferences. Has effect after Julia restart.

Restart Julia and verify it by `!isnothing(RxInfer.default_session())`.     
"""
function enable_session_logging!()
    @set_preferences!("enable_session_logging" => true)
    @info "Enabled session logging. Changes will take effect after Julia restart."
end

"""
    summarize_session([io::IO], session::Session, label::Symbol = :inference; n_last = 5)

Print a concise summary of session statistics for invokes with the specified label.
The default label is `:inference` which gathers statistics of the `infer` function calls.
"""
summarize_session(session::Session = RxInfer.default_session(), label::Symbol = :inference; n_last = 5) = summarize_session(stdout, session, label; n_last = n_last)

function summarize_session(io::IO, session::Union{Session, Nothing} = RxInfer.default_session(), label::Symbol = :inference; n_last = 5)
    if isnothing(session)
        println(io, "Session logging is disabled")
    end

    stats = get_session_stats(session, label)
    filtered_invokes = filter(i -> i.label === label, session.invokes)

    println(io, "\nSession Summary (label: $label)")
    println(io, "Total invokes: $(stats.total_invokes)")
    println(io, "Success rate: $(round(stats.success_rate * 100, digits=1))%")
    println(io, "Failed invokes: $(stats.failed_invokes)")
    println(io, "\nExecution time (ms):")
    println(io, "  Mean: $(round(stats.mean_duration_ms, digits=2))")
    println(io, "  Min: $(round(stats.min_duration_ms, digits=2))")
    println(io, "  Max: $(round(stats.max_duration_ms, digits=2))")
    println(io, "\nContext keys: $(join(stats.context_keys, ", "))")

    if stats.total_invokes == 0
        println(io, "\nNo invokes found with label: $label")
        return nothing
    end

    # Call label-specific summary with n_last parameter
    summarize_invokes(io, Val(label), filtered_invokes; n_last = n_last)

    return nothing
end

"""
    get_session_stats(session::Session, label::Symbol = :inference)

Return a NamedTuple with key session statistics for invokes with the specified label.

# Returns
- `total_invokes`: Total number of invokes with the given label
- `success_rate`: Fraction of successful invokes (between 0 and 1)
- `failed_invokes`: Number of failed invokes
- `mean_duration_ms`: Mean execution time in milliseconds
- `min_duration_ms`: Minimum execution time in milliseconds
- `max_duration_ms`: Maximum execution time in milliseconds
- `context_keys`: Set of all context keys used across invokes
- `label`: The label used for filtering
"""
function get_session_stats(session::Union{Nothing, Session} = RxInfer.default_session(), label::Symbol = :inference)
    empty_session = (total_invokes = 0, success_rate = 0.0, failed_invokes = 0, context_keys = Symbol[], label = label)

    if isnothing(session)
        return empty_session
    end

    filtered_invokes = filter(i -> i.label === label, session.invokes)
    n_invokes = length(filtered_invokes)

    if n_invokes == 0
        return empty_session
    end

    n_success = count(i -> i.status === :success, filtered_invokes)
    n_failed = count(i -> i.status === :failed, filtered_invokes)

    durations = map(filtered_invokes) do invoke
        Dates.value(invoke.execution_end - invoke.execution_start) / 1000.0
    end

    context_keys = unique(Iterators.flatten(keys(i.context) for i in filtered_invokes))

    stats = (
        total_invokes = n_invokes,
        success_rate = n_success / n_invokes,
        failed_invokes = n_failed,
        mean_duration_ms = mean(durations),
        min_duration_ms = minimum(durations),
        max_duration_ms = maximum(durations),
        context_keys = collect(context_keys),
        label = label
    )

    return stats
end
