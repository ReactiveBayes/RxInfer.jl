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
        # Firestore requires collection name in the URL
        collection = "using_rxinfer"
        endpoint = joinpath(preference_telemetry_endpoint, collection)

        # Headers required for Firestore REST API
        headers = ["Accept" => "application/json", "Content-Type" => "application/json"]

        # Firestore document structure
        # See: https://firebase.google.com/docs/firestore/reference/rest/v1/projects.databases.documents
        payload = Dict(
            "fields" => Dict("timestamp" => Dict("timestampValue" => Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SS.sssZ")), "id" => Dict("stringValue" => string(uuid4())))
        )

        # Make request to Firestore REST API
        HTTP.post(endpoint, headers, JSON.json(payload))
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
