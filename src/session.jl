using Dates, UUIDs, Preferences

struct InferInvoke
end

"""
    Session

A structure that maintains a log of all inference invocations during a RxInfer session.
Each session has a unique identifier and tracks when it was created. The session stores
a history of all inference invocations (`InferInvoke`) that occurred during its lifetime.

# Fields
- `id::UUID`: A unique identifier for the session
- `created_at::DateTime`: Timestamp when the session was created
- `invokes::Vector{InferInvoke}`: List of all inference invocations that occurred during the session

The session logging is transparent and only collects non-sensitive information about inference calls.
Users can inspect the session at any time using `get_current_session()` and reset it using `reset_session!()`.
"""
struct Session
    id::UUID
    created_at::DateTime
    invokes::Vector{InferInvoke}
end

"""
    create_session()

Create a new session with a unique identifier and current timestamp.

# Returns
- `Session`: A new session instance with no inference invocations recorded
"""
function create_session()
    return Session(
        uuid4(), # Generate unique ID
        now(),  # Current timestamp
        InferInvoke[] # Empty vector of invokes
    )
end

session_logging_preference = @load_preference("enable_session_logging", true)

const default_session_sem = Base.Semaphore(1)

# See `Preferences.jl` to see how it works, but it should be a compile time choice of a user
@static if session_logging_preference
const default_session_ref = Ref{Union{Nothing, Session}}(create_session())
else
const default_session_ref = Ref{Union{Nothing, Session}}(nothing) 
end

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