using Dates, UUIDs, Preferences

struct InferInvokeDataEntry
    name
    type
    size
    elsize
end

# Very safe by default, logging should not crash if we don't know how to parse the data entry
log_data_entry(data) = InferInvokeDataEntry(:unknown, :unknown, :unknown, :unknown)
log_data_entry(data::Pair) = log_data_entry(first(data), last(data))

log_data_entry(name::Union{Symbol, String}, data) = log_data_entry(name, Base.IteratorSize(data), data)
log_data_entry(name::Union{Symbol, String}, _, data) = InferInvokeDataEntry(name, typeof(data), :unknown, :unknown)
log_data_entry(name::Union{Symbol, String}, ::Base.HasShape{0}, data) = InferInvokeDataEntry(name, typeof(data), (), ())
log_data_entry(name::Union{Symbol, String}, ::Base.HasShape, data) = InferInvokeDataEntry(name, typeof(data), size(data), isempty(data) ? () : size(first(data)))

# Julia has `Base.HasLength` by default, which is quite bad because it fallbacks here 
# for structures that has nothing to do with being iterators nor implement `length`, 
# Better to be safe here and simply return :unknown
log_data_entry(name::Union{Symbol, String}, ::Base.HasLength, data) = InferInvokeDataEntry(name, typeof(data), :unknown, :unknown)

# Very safe by default, logging should not crash if we don't know how to parse the data entry
log_data_entries(data) = :unknown

log_data_entries(data::Union{NamedTuple, Dict}) = log_data_entries_from_pairs(pairs(data))
log_data_entries_from_pairs(pairs) = collect(Iterators.map(log_data_entry, pairs))

struct InferInvoke 
    id::UUID
    status::Symbol
    execution_start::DateTime
    execution_end::DateTime
    model
    data
end

"""
    Session

A structure that maintains a log of all inference invocations during a RxInfer session.
Each session has a unique identifier and tracks when it was created. The session stores
a history of all inference invocations (`InferInvoke`) that occurred during its lifetime.

# Fields
- `id::UUID`: A unique identifier for the session
- `created_at::DateTime`: Timestamp when the session was created
- `environment::Dict{Symbol, Any}`: Information about the Julia & RxInfer versions and system when the session was created
- `invokes::Vector{InferInvoke}`: List of all inference invocations that occurred during the session

The session logging is transparent and only collects non-sensitive information about inference calls.
Users can inspect the session at any time using `get_current_session()` and reset it using `reset_session!()`.
"""
struct Session
    id::UUID
    created_at::DateTime
    environment::Dict{Symbol, Any}
    invokes::Vector{InferInvoke}
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
        InferInvoke[]     # Empty vector of invokes
    )
end

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