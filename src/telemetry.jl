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
            endpoint = string(rstrip(preference_telemetry_endpoint, '/'), '/', collection, '/', id_name_mapping[id])
            HTTP.patch(endpoint, headers, JSON.json(payload))
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