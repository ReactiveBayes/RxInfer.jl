# RxInfer Usage Telemetry

## [Usage Telemetry](@id manual-usage-telemetry)

RxInfer collects minimal anonymous usage statistics to help improve the package. The only data collected is:
- A timestamp of when the package is loaded
- A random UUID for deduplication

These statistics help us understand how RxInfer is being used and guide our development efforts. We share and discuss these numbers in our public meetings, which happen every 4 weeks. We encourage you to join our community meetings to:
- Learn about usage patterns and how others are using RxInfer
- Provide feedback and suggestions
- Help shape the future of the package
- Connect with the RxInfer community

### Join Our Meetings

We hold public meetings every 4 weeks where we:
- Share and discuss usage statistics
- Present new features and improvements
- Discuss the roadmap and future development
- Answer questions from the community
- Welcome feedback and suggestions

To join:
1. Check our [meeting schedule](https://dynalist.io/d/F4aA-Z2c8X-M1iWTn9hY_ndN) for the next meeting date
2. Join our [GitHub discussions](https://github.com/reactivebayes/RxInfer.jl/discussions) to stay updated
3. Feel free to add topics you'd like to discuss to the meeting agenda

We welcome participants of all experience levels - whether you're new to RxInfer or a seasoned user!

### Default Behavior

By default, [Usage Telemetry](@ref manual-usage-telemetry) is enabled but you can disable it in several ways:

1. Using environment variables:
   ```bash
   export LOG_USING_RXINFER=false
   ```

2. Using Julia functions:
   ```julia
   using RxInfer
   RxInfer.disable_rxinfer_using_telemetry!() # Requires Julia restart
   ```

3. Setting the endpoint to `nothing`:
   ```julia
   using RxInfer
   RxInfer.set_telemetry_endpoint!(nothing) # Requires Julia restart
   ```

### Configuration Functions

The following functions are available for telemetry configuration:

- `set_telemetry_endpoint!(endpoint)`: Set a custom telemetry endpoint or disable telemetry by setting it to `nothing`
- `disable_rxinfer_using_telemetry!()`: Disable telemetry collection (requires Julia restart)
- `enable_rxinfer_using_telemetry!()`: Enable telemetry collection (requires Julia restart)

### When Telemetry is Disabled

Telemetry is automatically disabled in the following cases:
1. When running in CI environments (detected via `CI=true` environment variable)
2. When `LOG_USING_RXINFER=false` environment variable is set
3. When telemetry is disabled via `disable_rxinfer_using_telemetry!()`
4. When the telemetry endpoint is set to `nothing`

### Privacy Considerations

- No personal information is collected
- No code from your sessions is transmitted, but you can share your sessions if you want to help us to debug issues and improve the package. Read more about [session sharing](@ref manual-session-sharing)
- All requests are made asynchronously and will never block or affect your work
- Failed telemetry requests are silently discarded
- The code is open source and can be inspected in the `src/telemetry.jl` file
- Usage statistics are shared during our public meetings

## [Session Sharing](@id manual-session-sharing)

Please, first read the [Session Summary](@ref manual-session-summary) manual to understand the basic concepts of RxInfer sessions.