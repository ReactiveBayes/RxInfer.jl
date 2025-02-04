const RXINFER_DEFAULT_TELEMETRY_ENDPOINT = "https://firestore.googleapis.com/v1/projects/reactive-bayes/databases/(default)/documents/"

preference_telemetry_endpoint = @load_preference("telemetry_endpoint", RXINFER_DEFAULT_TELEMETRY_ENDPOINT)

preference_enable_using_rxinfer_telemetry = @load_preference("enable_using_rxinfer_telemetry", true)

"""
    set_telemetry_endpoint!(endpoint)

Set the telemetry endpoint URL for RxInfer.jl at compile time. This endpoint is used for collecting anonymous usage statistics
to help improve the package.

The change requires a Julia session restart to take effect.

# Arguments
- `endpoint`: The URL of the telemetry endpoint as a `String` or `nothing``
"""
function set_telemetry_endpoint!(endpoint)
    @set_preferences!("telemetry_endpoint" => endpoint)
    if !isnothing(endpoint)
        @info "Telemetry endpoint set to $endpoint. Restart Julia for the change to take effect."
    elseif isnothing(endpoint)
        @info "Telemetry endpoint is set to `nothing`."
    end
    return nothing
end

using HTTP, Dates, UUIDs, JSON

const logged_usage = Ref(false)

"""
    log_using_rxinfer()

Send an anonymous usage statistics event to the telemetry endpoint on `using RxInfer`.
This function makes an asynchronous HTTP POST request to the configured endpoint.
See `RxInfer.set_telemetry_endpoint!` to configure the endpoint.
If the telemetry endpoint is set to `nothing`, this function does nothing.
The call sends only timestamp and a random UUID. 
The request is made asynchronously to avoid blocking the user's workflow.
See `RxInfer.disable_rxinfer_using_telemetry!` to disable telemetry on `using RxInfer`.
Alternatively, set the environment variable `LOG_USING_RXINFER` to `false` to disable logging.
"""
function log_using_rxinfer()
    if logged_usage[]
        return nothing
    end

    if !preference_enable_using_rxinfer_telemetry || isnothing(preference_telemetry_endpoint)
        return nothing
    end

    # Do not log usage statistics in CI
    if get(ENV, "CI", "false") == "true"
        return nothing
    end

    # Do not log usage statistics if the environment variable is set to `false`
    if get(ENV, "LOG_USING_RXINFER", "true") == "false"
        return nothing
    end

    Base.Threads.@spawn :interactive try
        id = string(uuid4())
        __add_document(id, "using_rxinfer", (fields = (timestamp = (timestampValue = Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),), id = (stringValue = id,)),))
    catch e
        nothing # Discard any log errors to avoid disrupting the user
    end

    logged_usage[] = true

    return nothing
end

"""
    disable_rxinfer_using_telemetry!()

Disable telemetry collection on `using RxInfer` at compile time. The change requires a Julia session restart to take effect.

See also: [`set_telemetry_endpoint!`](@ref), [`enable_rxinfer_using_telemetry!`](@ref)
"""
function disable_rxinfer_using_telemetry!()
    @set_preferences!("enable_using_rxinfer_telemetry" => false)
    @info "Disabled telemetry collection on `using RxInfer`. Changes will take effect after Julia restart."
    return nothing
end

"""
    enable_rxinfer_using_telemetry!()

Enable telemetry collection on `using RxInfer` at compile time. The change requires a Julia session restart to take effect.

See also: [`set_telemetry_endpoint!`](@ref), [`disable_rxinfer_using_telemetry!`](@ref)
"""
function enable_rxinfer_using_telemetry!()
    @set_preferences!("enable_using_rxinfer_telemetry" => true)
    @info "Enabled telemetry collection on `using RxInfer`. Changes will take effect after Julia restart."
    return nothing
end

# The mapping of the document ID to the endpoint name
# This is used to avoid duplicate documents in Firestore
const id_name_mapping = Dict{String, String}()

# The mapping of the collection name to the allow_patch flag
# This is used to avoid pushing data to Firestore if the document already exists
const collection_allow_patch = Dict{String, Bool}("using_rxinfer" => true, "sessions" => true, "session_stats" => true, "invokes" => false)

# Adds or updates a document in Firestore based on the provided id and collection.
# If a document with the same id already exists (tracked in id_name_mapping), 
# it updates that document instead of creating a new one to avoid duplicates.
# The document name from Firestore is stored in id_name_mapping for future updates.
function __add_document(id, collection, payload)
    if !isnothing(preference_telemetry_endpoint)
        # Headers required for Firestore REST API
        headers = ["Accept" => "application/json", "Content-Type" => "application/json"]

        # Example values:
        # id = "550e8400-e29b-41d4-a716-446655440000" (a UUID string)
        # collection = "using_rxinfer"
        # payload = (fields = (
        #     timestamp = (timestampValue = "2024-01-20T14:30:15.123Z"),
        #     id = (stringValue = "550e8400-e29b-41d4-a716-446655440000")
        # ))
        # preference_telemetry_endpoint = "https://firestore.googleapis.com/v1/projects/myproject/databases/(default)/documents"

        # Firestore document structure
        # See: https://firebase.google.com/docs/firestore/reference/rest/v1/projects.databases.documents
        response = if haskey(id_name_mapping, id)
            # If document exists, endpoint would be like:
            # "https://firestore.../using_rxinfer/abc123def456"
            name = id_name_mapping[id]
            endpoint = string(rstrip(preference_telemetry_endpoint, '/'), '/', collection, '/', name)
            # For collections that allow patching (like using_rxinfer, sessions, session_stats),
            # send a PATCH request to update the existing document with new data
            if collection_allow_patch[collection]
                HTTP.patch(endpoint, headers, JSON.json(payload))
            # For collections that don't allow patching (like invokes),
            # return a fake successful response without making a request,
            # since we don't want to update existing documents in these collections
            else
                (body = """{"name": "$name"}""", status = 200)
            end
        else
            # For new documents, endpoint would be like:
            # "https://firestore.../using_rxinfer"
            endpoint = string(rstrip(preference_telemetry_endpoint, '/'), '/', collection)
            HTTP.post(endpoint, headers, JSON.json(payload))
        end

        # Parse response if successful
        # Example response body:
        # {
        #   "name": "projects/myproject/databases/(default)/documents/using_rxinfer/abc123def456",
        #   "fields": { ... },
        #   "createTime": "2024-01-20T14:30:15.123456Z",
        #   "updateTime": "2024-01-20T14:30:15.123456Z"
        # }
        if response.status == 200
            body = JSON.parse(String(response.body))
            name = get(body, "name", nothing)
            if !isnothing(name)
                # Extract just the document ID ("abc123def456") from the full path
                name = split(name, "/") |> last
                id_name_mapping[id] = name
            end
            return name
        end
    end
    return nothing
end

# Conversion functions between Julia objects and Firestore data types

"""
    to_firestore_value(value)

Convert a Julia value to a Firestore-compatible value format.
Returns a NamedTuple with the appropriate Firestore field type.
"""
function to_firestore_value(value::String)
    return (stringValue = value,)
end

function to_firestore_value(value::Symbol)
    return (stringValue = string(value),)
end

function to_firestore_value(value::Integer)
    return (integerValue = value,)
end

function to_firestore_value(value::AbstractFloat)
    return (doubleValue = value,)
end

function to_firestore_value(value::DateTime)
    return (timestampValue = Dates.format(value, "yyyy-mm-ddTHH:MM:SS.sssZ"),)
end

function to_firestore_value(value::UUID)
    return (stringValue = string(value),)
end

function to_firestore_value(value::Bool)
    return (booleanValue = value,)
end

function to_firestore_value(value::Nothing)
    return (nullValue = nothing,)
end

function to_firestore_value(value::AbstractVector)
    return (arrayValue = (values = [to_firestore_value(item) for item in value],),)
end

function to_firestore_value(value::AbstractDict)
    return (mapValue = (fields = Dict(string(k) => to_firestore_value(v) for (k, v) in value),),)
end

function to_firestore_value(value::Any)
    # For any other type, convert to string but include the type information
    return (stringValue = string("$(typeof(value)): ", value),)
end

"""
    to_firestore_document(data::NamedTuple)

Convert a Julia NamedTuple to a Firestore document format.
Returns a NamedTuple with fields in Firestore format.
"""
function to_firestore_document(data::NamedTuple)
    return (fields = NamedTuple(k => to_firestore_value(v) for (k, v) in pairs(data)),)
end

"""
    to_firestore_session(session::Session)

Convert a Session object to a Firestore-compatible document format.
"""
function to_firestore_session(session::Session)
    return to_firestore_document((id = session.id, created_at = session.created_at, environment = session.environment))
end

"""
    to_firestore_session_stats(stats::SessionStats, session_id::UUID)

Convert a SessionStats object to a Firestore-compatible document format.
Includes a reference to the parent session.
"""
function to_firestore_session_stats(stats::SessionStats, session_id::UUID)
    return to_firestore_document((
        id = stats.id,
        session = session_id,
        label = stats.label,
        total_invokes = stats.total_invokes,
        success_count = stats.success_count,
        failed_count = stats.failed_count,
        success_rate = stats.success_rate,
        min_duration_ms = stats.min_duration_ms == Inf ? 0.0 : stats.min_duration_ms,
        max_duration_ms = stats.max_duration_ms == -Inf ? 0.0 : stats.max_duration_ms,
        total_duration_ms = stats.total_duration_ms,
        context_keys = collect(stats.context_keys)
    ))
end

"""
    to_firestore_invoke(invoke::SessionInvoke, stats_id::UUID)

Convert a SessionInvoke object to a Firestore-compatible document format.
Includes a reference to the parent session stats.
"""
function to_firestore_invoke(invoke::SessionInvoke, stats_id::UUID)
    return to_firestore_document((
        id = invoke.id, session_stats = stats_id, status = invoke.status, execution_start = invoke.execution_start, execution_end = invoke.execution_end, context = invoke.context
    ))
end

import ProgressMeter

"""
    share_session_data(session = RxInfer.default_session(); show_progress::Bool = true)

Share your session data to help improve RxInfer.jl and its community. This data helps us:
- Understand how the package is used in practice
- Identify areas for improvement
- Make informed decisions about future development
- Share aggregate usage patterns in our community meetings

The data is organized in a structured way:
1. Basic session info (Julia version, OS, etc.)
2. Anonymous statistics about different types of package usage
3. Information about individual labeled runs

All data is anonymous and only used to improve the package. We discuss aggregate statistics 
in our public community meetings to make the development process transparent and collaborative.

# Arguments
- `session::Session`: The session object containing data to share
- `show_progress::Bool = true`: Whether to display progress bars during sharing

# Progress Display
When `show_progress` is true (default), the function displays:
- A blue progress bar for sharing session statistics
- A green progress bar for sharing labeled runs
"""
function share_session_data(session::Union{Session, Nothing} = RxInfer.default_session(); show_progress::Bool = true)
    if isnothing(preference_telemetry_endpoint)
        @warn "Cannot share session data: telemetry endpoint is not set. See `RxInfer.set_telemetry_endpoint!()`"
        return nothing
    end

    if isnothing(session)
        @warn "Cannot share session data: session logging is not enabled. See `RxInfer.enable_session_logging!()`"
        return nothing
    end

    @info "Starting to share session data to help improve RxInfer.jl" session_id = session.id num_stats = length(session.stats)

    # Share session information
    session_name = __add_document(string(session.id), "sessions", to_firestore_session(session))
    if isnothing(session_name)
        @warn "Unable to share session data" session_id = session.id
        return nothing
    end

    # Track sharing progress
    total_stats = length(session.stats)
    shared_stats = 0
    total_invokes = sum(length(stats.invokes) for (_, stats) in session.stats)
    shared_invokes = 0

    # Create progress meter for stats if requested
    stats_progress = show_progress ? ProgressMeter.Progress(total_stats; desc = "Sharing statistics: ", color = :blue) : nothing

    # Share session statistics
    for (label, stats) in session.stats
        stats_name = __add_document(string(stats.id), "session_stats", to_firestore_session_stats(stats, session.id))
        if isnothing(stats_name)
            @warn "Unable to share statistics data" stats_id = stats.id label = label session_id = session.id
            continue
        end
        shared_stats += 1
        show_progress && ProgressMeter.next!(stats_progress)

        # Create progress meter for invokes within this stats if requested
        invokes_progress = show_progress ? ProgressMeter.Progress(length(stats.invokes); desc = "Sharing runs for $(label): ", color = :green) : nothing

        # Share labeled runs
        invokes_shared = 0
        for invoke in stats.invokes
            invoke_name = __add_document(string(invoke.id), "invokes", to_firestore_invoke(invoke, stats.id))
            if isnothing(invoke_name)
                @warn "Unable to share run data" invoke_id = invoke.id stats_id = stats.id session_id = session.id
                continue
            end
            invokes_shared += 1
            shared_invokes += 1
            show_progress && ProgressMeter.next!(invokes_progress)
        end

        if invokes_shared < length(stats.invokes)
            @warn "Some runs could not be shared" stats_id = stats.id label = label shared = invokes_shared total = length(stats.invokes) session_id = session.id
        end
    end

    # Final summary
    if shared_stats < total_stats || shared_invokes < total_invokes
        @warn """
        Session data sharing completed with some limitations.
        We appreciate your contribution even if not all data could be shared.
        """ session_id = session.id shared_stats = shared_stats total_stats = total_stats shared_invokes = shared_invokes total_invokes = total_invokes
    else
        @info """
        Thank you for sharing your session data! 

        This helps us understand how RxInfer.jl is used and guides our improvements.
        We discuss aggregate usage patterns in our public community meetings to make
        the development process transparent and collaborative.

        When opening issues on GitHub at https://github.com/reactivebayes/RxInfer.jl/issues/new, 
        please include this session ID `$(session.id)` and session name: `$(session_name)`.

        Optionally, provide IDs of individual runs that you are interested in.
        Call `RxInfer.summarize_session()` to get the list of run IDs.

        This helps us provide better support by understanding your usage context.
        """ session_id = session.id session_name = session_name stats_count = shared_stats invokes_count = shared_invokes
    end

    return nothing
end
