using Dates, UUIDs, Preferences

import DataStructures: CircularBuffer, capacity

preference_enable_session_logging = @load_preference("enable_session_logging", true)

"""
    SessionInvoke

Represents a single invocation of an inference operation.

# Fields
- `id::UUID`: Unique identifier for this invocation
- `status::Symbol`: Status of the invocation (e.g. :success, :failure)
- `execution_start::DateTime`: When the invocation started
- `execution_end::DateTime`: When the invocation completed
- `context::Dict{Symbol, Any}`: Additional contextual information
"""
mutable struct SessionInvoke
    id::UUID
    status::Symbol
    execution_start::DateTime
    execution_end::DateTime
    context::Dict{Symbol, Any}
end

"""
    SessionStats

Statistics for a specific label in a session.

# Fields
- `id::UUID`: Unique identifier for these statistics
- `label::Symbol`: The label these statistics are for
- `total_invokes::Int`: Total number of invokes with this label
- `success_count::Int`: Number of successful invokes
- `failed_count::Int`: Number of failed invokes
- `success_rate::Float64`: Fraction of successful invokes (between 0 and 1)
- `min_duration_ms::Float64`: Minimum execution duration in milliseconds
- `max_duration_ms::Float64`: Maximum execution duration in milliseconds
- `total_duration_ms::Float64`: Total execution duration for mean calculation
- `context_keys::Set{Symbol}`: Set of all context keys used across invokes
- `invokes::CircularBuffer{SessionInvoke}`: A series of invokes attached to the statistics
"""
mutable struct SessionStats
    id::UUID
    label::Symbol
    total_invokes::Int
    success_count::Int
    failed_count::Int
    success_rate::Float64
    min_duration_ms::Float64
    max_duration_ms::Float64
    total_duration_ms::Float64
    context_keys::Set{Symbol}
    invokes::CircularBuffer{SessionInvoke}
end

"""
    DEFAULT_SESSION_STATS_CAPACITY

The default capacity for the circular buffer storing session invocations.
This value determines how many past invocations are stored for each label's statistics.
Can be modified at compile time using preferences:

```julia
using RxInfer
set_session_stats_capacity!(100)
```

The change requires a Julia session restart to take effect. Default value is `1000`.
Must be a positive integer.
"""
const DEFAULT_SESSION_STATS_CAPACITY = @load_preference("session_stats_capacity", 1000)

"""
    set_session_stats_capacity!(capacity::Int)

Set the default capacity for session statistics at compile time. The change requires a Julia session restart to take effect.
"""
function set_session_stats_capacity!(capacity::Int)
    @assert capacity > 0 "Session stats capacity must be positive"
    @set_preferences!("session_stats_capacity" => capacity)
    @info "Session stats capacity set to $capacity. Restart Julia for the change to take effect."
end

# Constructor for empty stats
function SessionStats(label::Symbol, capacity::Int = DEFAULT_SESSION_STATS_CAPACITY)
    invokes = CircularBuffer{SessionInvoke}(capacity)
    return SessionStats(uuid4(), label, 0, 0, 0, 0.0, Inf, -Inf, 0.0, Set{Symbol}(), invokes)
end

"""
    Session

A structure that maintains a log of RxInfer usage.
Each session has a unique identifier and saves when it was created together with its environment. 
The session maintains a dictionary of labeled statistics, each tracking a series of invocations (`SessionInvoke`) 
and computing real-time statistics.

# Fields
- `id::UUID`: A unique identifier for the session
- `created_at::DateTime`: Timestamp when the session was created
- `environment::Dict{Symbol, Any}`: Information about the Julia & RxInfer versions and system when the session was created
- `semaphore::Base.Semaphore`: Thread-safe semaphore for updating stats
- `stats::Dict{Symbol, SessionStats}`: Statistics per label

The session logging is transparent and only collects non-sensitive information about calls.
Users can inspect the session at any time using `get_current_session()` and reset it using `reset_session!()`.
"""
struct Session
    id::UUID
    created_at::DateTime
    environment::Dict{Symbol, Any}
    semaphore::Base.Semaphore
    stats::Dict{Symbol, SessionStats}
end

"""
    create_session()

Create a new session with a unique identifier, environment info and current timestamp.
The session maintains separate statistics for each label, with each label's statistics
having its own circular buffer of invokes.
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
        Base.Semaphore(1),  # Thread-safe semaphore
        Dict{Symbol, SessionStats}()  # Empty stats dictionary
    )
end

"""
    reset_session!(session, [ labels ])

Removes gathered statistics from the session. Optionally accepts a vector of labels to delete.
If no labels specified deletes everything.
"""
function reset_session!(session::Union{Nothing, Session} = RxInfer.default_session(), labels = nothing)
    if isnothing(labels)
        labels = keys(session.stats)
    end
    for label in labels
        if haskey(session.stats, label)
            delete!(session.stats, label)
            @info "Removed statistics for `$label`"
        else
            @warn "Cannot remove statistics for `$label`. Statistics labeled with `$label` do not exist."
        end
    end
end

"""
    create_invoke()

Create a new session invoke with status set to `:unknown`.
"""
function create_invoke()
    return SessionInvoke(uuid4(), :unknown, Dates.now(), Dates.now(), Dict{Symbol, Any}())
end

"""
    update_stats!(stats::SessionStats, invoke::SessionInvoke)

Update session statistics with a new invoke.
"""
function update_stats!(stats::SessionStats, invoke::SessionInvoke)
    stats.total_invokes += 1

    # Update success/failure counts
    if invoke.status === :success
        stats.success_count += 1
    elseif invoke.status === :error
        stats.failed_count += 1
    end

    # Update success rate
    stats.success_rate = stats.success_count / stats.total_invokes

    # Calculate duration in milliseconds
    duration_ms = Dates.value(Dates.Millisecond(invoke.execution_end - invoke.execution_start))

    # Update duration stats
    stats.min_duration_ms = min(stats.min_duration_ms, duration_ms)
    stats.max_duration_ms = max(stats.max_duration_ms, duration_ms)
    stats.total_duration_ms += duration_ms

    # Update context keys
    union!(stats.context_keys, keys(invoke.context))

    push!(stats.invokes, invoke)
end

"""
    update_session!(session::Session, label::Symbol, invoke::SessionInvoke)

Thread-safely update session statistics for a given label with a new invoke.
Uses a semaphore to ensure thread safety when multiple threads try to update statistics simultaneously.

# Arguments
- `session::Session`: The session to update
- `label::Symbol`: Label for the invoke
- `invoke::SessionInvoke`: The invoke to add to statistics
"""
function update_session!(session::Session, label::Symbol, invoke::SessionInvoke)
    return Base.acquire(session.semaphore) do
        # Get or create stats for this label
        stats = get!(session.stats, label) do
            SessionStats(label)
        end

        # Update stats with new invoke
        update_stats!(stats, invoke)
    end
end

"""
    with_session(f::F, session, label::Symbol = :unknown) where {F}

Execute function `f` within a session context with the specified label. If `session` is provided,
logs execution details including timing and errors, and updates the session statistics for the given label.
If `session` is `nothing`, executes `f` without logging.
"""
function with_session(f::F, session, label::Symbol) where {F}
    if isnothing(session)
        return f(nothing)
    elseif session isa Session
        invoke = create_invoke()
        try
            result = f(invoke)
            invoke.status = :success
            invoke.execution_end = Dates.now()
            return result
        catch e
            invoke.status = :error
            invoke.context[:error] = string(e)
            rethrow(e)
        finally
            update_session!(session, label, invoke)
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
        return nothing
    end

    stats   = get_session_stats(session, label)
    invokes = stats.invokes

    println(io, "\nSession Summary (label: $label)")
    println(io, "Total invokes: $(stats.total_invokes)")
    println(io, "Session invokes limit: $(capacity(invokes))")
    println(io, "Success rate: $(round(stats.success_rate * 100, digits=1))%")
    println(io, "Failed invokes: $(stats.failed_count)")

    mean_execution = round(stats.total_duration_ms / max(1, stats.total_invokes), digits=2)
    min_execution = stats.min_duration_ms == Inf ? 0.0 : round(stats.min_duration_ms, digits=2)
    max_execution = stats.max_duration_ms == -Inf ? 0.0 : round(stats.max_duration_ms, digits=2)

    println(io, "Average execution time ", mean_execution, "ms (min: ", min_execution, "ms, max: ", max_execution, "ms)")
    println(io, "Context keys: $(join(collect(stats.context_keys), ", "))")

    if stats.total_invokes == 0
        println(io, "\nNo invokes found with label: $label")
        return nothing
    end

    # Call label-specific summary with n_last parameter
    summarize_invokes(io, Val(label), invokes; n_last = n_last)

    return nothing
end

"""
    get_session_stats(session::Session, label::Symbol = :inference)

Get statistics for invokes with the specified label. If the label doesn't exist in the session,
returns a new empty `SessionStats` instance.

# Arguments
- `session::Union{Nothing, Session}`: The session to get statistics from, or nothing
- `label::Symbol`: The label to get statistics for, defaults to :inference

"""
function get_session_stats(session::Union{Nothing, Session} = RxInfer.default_session(), label::Symbol = :inference)
    if isnothing(session)
        return SessionStats(label)
    end
    return Base.acquire(session.semaphore) do
        return get!(session.stats, label, SessionStats(label))
    end
end

# Show methods for nice printing
function Base.show(io::IO, invoke::SessionInvoke)
    duration_ms = round(Dates.value(Dates.Millisecond(invoke.execution_end - invoke.execution_start)), digits=2)
    print(io, "SessionInvoke(id=$(invoke.id), status=$(invoke.status), duration=$(duration_ms)ms, context_keys=[$(join(keys(invoke.context), ", "))])")
end

function Base.show(io::IO, stats::SessionStats)
    print(io, "SessionStats(id=$(stats.id), label=:$(stats.label), total=$(stats.total_invokes), success_rate=$(round(stats.success_rate * 100, digits=1))%, invokes=$(length(stats.invokes))/$(capacity(stats.invokes)))")
end

function Base.show(io::IO, session::Session)
    print(io, "Session(id=$(session.id), labels=[$(join(keys(session.stats), ", "))])")
end