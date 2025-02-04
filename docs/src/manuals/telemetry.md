# RxInfer Usage Telemetry

## [Usage Telemetry](@id manual-usage-telemetry)

RxInfer includes an optional telemetry system that can help us understand how the package is used and guide our improvements. 

```@docs
RxInfer.log_using_rxinfer
```

### What We Collect

We collect minimal anonymous usage statistics:
- A timestamp of when the package is loaded
- A random UUID for deduplication

### How This Helps

This anonymous data helps us:
- Understand how RxInfer is used in practice
- Make informed decisions about future development
- Share aggregate usage patterns in our community meetings

### Community Transparency

We believe in full transparency about how we use this data:
- We discuss aggregate statistics in our public meetings (every 4 weeks)
- All telemetry code is open source and can be inspected in `src/telemetry.jl`
- Failed telemetry requests are silently discarded
- All requests are asynchronous and never block your work

### How to Enable/Disable Telemetry

You can interact with telemetry in several ways:

1. Using Julia functions:
   ```julia
   using RxInfer
   RxInfer.enable_rxinfer_using_telemetry!() # Requires Julia restart
   ```

2. Or disable it at any time:
   ```julia
   using RxInfer
   RxInfer.disable_rxinfer_using_telemetry!() # Requires Julia restart
   ```

3. Using environment variables:
   ```bash
   # To disable
   export LOG_USING_RXINFER=false
   
   # To enable (default)
   export LOG_USING_RXINFER=true
   ```

### When Telemetry is Automatically Disabled

Telemetry is automatically disabled in the following cases:
1. When running in CI environments (detected via `CI=true` environment variable)
2. When `LOG_USING_RXINFER=false` environment variable is set
3. When telemetry is disabled via `disable_rxinfer_using_telemetry!()`
4. When the telemetry endpoint is set to `nothing`

```@docs 
RxInfer.disable_rxinfer_using_telemetry!
RxInfer.enable_rxinfer_using_telemetry!
RxInfer.set_telemetry_endpoint!
```

## [Session Sharing](@id manual-session-sharing)

For more complex scenarios, like debugging issues or getting help from the community, you can choose to share your session data. This is separate from telemetry and gives us more context to help solve problems.

### What Session Data Contains

When you share a session, it includes:
- Basic session info (Julia version, OS, etc.)
- Anonymous statistics about different types of package usage
- Information about individual labeled runs
- No personal information or sensitive data

### How to Share Sessions

You can share your session data using the `share_session_data` function:

```@docs
RxInfer.share_session_data
```

### Using Session IDs in Issues

When you share a session and then open a GitHub issue, include your session ID. This helps us:
- Link your issue to the shared session data
- Understand your usage context
- Provide more accurate and helpful support

### Privacy and Control

Remember:
- Session sharing is completely optional
- All statistics are anonymous
- No actual data is shared, only meta information, e.g. type of the data, number of observations, etc.
- You can inspect the sharing code in `src/telemetry.jl`
- We only use this data to help improve RxInfer and provide better support

We appreciate your help in making RxInfer better! Whether you choose to enable telemetry or share sessions, your contribution helps us improve the package for everyone.

## Developers Reference 

```@docs
RxInfer.to_firestore_invoke
RxInfer.to_firestore_value
RxInfer.to_firestore_session
RxInfer.to_firestore_document
RxInfer.to_firestore_session_stats
```