using Dates, UUIDs, Preferences

mutable struct SessionInvoke 
    id::UUID
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
        :word_size => Sys.WORD_SIZE,
    )
    return Session(
        uuid4(),          # Generate unique ID
        now(),            # Current timestamp
        environment,      # Environment information
        SessionInvoke[]     # Empty vector of invokes
    )
end

"""
    with_session(f::F, session) where {F}

Execute function `f` within a session context. If `session` is provided, logs execution details including timing and errors.
If `session` is `nothing`, executes `f` without logging.
"""
function with_session(f::F, session) where {F}
    if isnothing(session)
        return f(nothing)
    elseif session isa Session
        invoke_id = uuid4()
        invoke_status = :unknown
        invoke_context = Dict{Symbol, Any}()
        invoke_execution_start = Dates.now()
        invoke_execution_end = Dates.now()
        invoke = SessionInvoke(
            invoke_id, 
            invoke_status,
            invoke_execution_start,
            invoke_execution_end,
            invoke_context
        )
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
end

"""
Enables session logging for RxInfer globally at compile time and saves it in package preferences. Has effect after Julia restart.

Restart Julia and verify it by `!isnothing(RxInfer.default_session())`.     
"""
function enable_session_logging!()
    @set_preferences!("enable_session_logging" => true)
end