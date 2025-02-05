# RxInfer Usage Telemetry

RxInfer has two separate telemetry features:
1. A minimal package usage counter (enabled by default)
2. Optional session sharing for better support and development insights

Users are welcome to join our regular online meetings where we analyze the collected data and discuss how it helps shape RxInfer's development.

## Package Usage Counter

By default, RxInfer counts how many times the package is loaded via `using RxInfer`. This counter:
- Only records a timestamp and a random UUID for deduplication
- UUIDs are not persistent and are re-generated for each session
- Does not collect any code, data, or environment information
- Is completely anonymous
- Helps us understand how widely RxInfer is used

### Disabling Package Usage Counter

You can disable the counter in several ways:

1. Using Julia functions:
   ```julia
   using RxInfer
   RxInfer.disable_rxinfer_using_telemetry!() # Requires Julia restart
   ```

2. Using environment variables:
   ```bash
   export LOG_USING_RXINFER=false
   ```

The counter is also automatically disabled in:
1. CI environments (detected via `CI=true` environment variable)
2. When telemetry is disabled via `disable_rxinfer_using_telemetry!()`
3. When the telemetry endpoint is set to `nothing`

```@docs 
RxInfer.log_using_rxinfer
RxInfer.disable_rxinfer_using_telemetry!
RxInfer.enable_rxinfer_using_telemetry!
RxInfer.set_telemetry_endpoint!
```

## [Session Sharing](@id manual-session-sharing)

RxInfer includes a built-in session tracking feature (detailed in [Session Summary](@ref manual-session-summary)) that helps you monitor and debug your inference tasks. You can choose to share these sessions with core developers to:
- Get better support when encountering issues
- Help improve RxInfer through real-world usage insights
- Contribute to community-driven development

Read more about what data is present in the session history in the [Session Summary](@ref manual-session-summary) manual.

### How to Share Sessions

You can share your session data either manually or automatically.

#### Manual Sharing

Use the `share_session_data` function to manually share your session:

```@docs
RxInfer.share_session_data
```

#### Automatic Sharing

You can enable automatic session sharing after each session update:

```@docs
RxInfer.enable_automatic_session_sharing!
RxInfer.disable_automatic_session_sharing!
```

When automatic sharing is enabled:
- Session data is shared after each session update
- Sharing is done asynchronously (won't block your code)
- No progress bars or messages are shown
- Failed sharing attempts are silently ignored

### Using Session IDs in Issues

When you share a session and then open a GitHub issue, include your session ID. This helps us:
- Link your issue to the shared session data
- Understand your usage context
- Provide more accurate and helpful support

### Deleting Shared Data

If you wish to delete previously shared session data, you can contact the core developers through GitHub issues at [RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl) or anonymously with the following [form](https://docs.google.com/forms/d/e/1FAIpQLSfLF5HcJODLyvovh0vOTjkh0b8it1GDUlyViqpDH06BxVbyYA/viewform?usp=sharing).

When requesting deletion, you must provide the session UUID. Without this identifier, we cannot trace specific sessions back to individual users. See the [Session Summary](@ref manual-session-summary) manual for details on how to obtain your session ID.

### Privacy and Control

Remember:
- Session sharing is completely optional
- All statistics are anonymous, UUIDs are not persistent and are re-generated for each session
- No actual data is shared, only meta information (e.g., type of data, number of observations)
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