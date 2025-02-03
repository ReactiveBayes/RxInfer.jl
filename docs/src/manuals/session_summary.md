# [Session Summary](@id manual-session-summary)

RxInfer provides a built-in session logging system that helps track and analyze various aspects of RxInfer usages. This feature is particularly useful for debugging, performance monitoring, and understanding the behavior of your inference models.

## Overview

Session logging in RxInfer automatically captures and maintains statistics for:
- Model source code and metadata
- Input data characteristics
- Execution timing and success rates
- Error information (if any)
- Environment information (Julia version, OS, etc.)
- Context keys used across invocations

## Basic Usage

By default, RxInfer creates and maintains a global session that logs all inference invocations:

```@example basic-session
using RxInfer

# Define a simple model
@model function simple_model(y)
    x ~ Normal(mean = 0.0, var = 1.0)
    y ~ Normal(mean = x, var = 1.0)
end

# Run inference with default session logging
result = infer(
    model = simple_model(),
    data = (y = 1.0,)
)

# Access the current session
session = RxInfer.default_session()

# Get statistics for inference invocations
stats = RxInfer.get_session_stats(session, :inference)
println("Number of invokes: $(stats.total_invokes)")
println("Success rate: $(round(stats.success_rate * 100, digits=1))%")
```

!!! note
    The number of logged invocations may be different from the number of invocations in the example above
    since the session is created and logged at the start of the documentation build.

## Session Capacity

By default, RxInfer maintains a fixed-size history of the invocations. 
When this limit is exceeded, the oldest invocations are automatically dropped. This prevents 
memory growth while maintaining recent history. 

```@docs
RxInfer.DEFAULT_SESSION_STATS_CAPACITY
RxInfer.set_session_stats_capacity!
```

This is particularly useful when:
- Running benchmarks that might generate many invocations
- Working with long-running applications
- Managing memory usage in resource-constrained environments

!!! note
    Changing the session stats capacity requires a Julia session restart to take effect.
    The change is persistent across Julia sessions until explicitly changed again.

## Custom sessions

You can also pass custom sessions to the `infer` function:

```@example custom-session
using RxInfer

@model function simple_model(y)
    x ~ Normal(mean = 0.0, var = 1.0)
    y ~ Normal(mean = x, var = 1.0)
end

# Create a custom session
session = RxInfer.create_session()

# Run inference with custom session
result = infer(
    model = simple_model(),
    data = (y = 1.0,),
    session = session
)

println("Session ID: $(session.id)")
println("Created at: $(session.created_at)")
```

or pass `nothing` to disable session logging:

```@example custom-session
result = infer(
    model = simple_model(),
    data = (y = 1.0,),
    session = nothing # skips session logging for this invocation
)
```

See [Configuration](@ref session-configuration) for more details on how to manage sessions.

## Session Reset 

You can reset the session to its initial state with [`RxInfer.reset_session!`](@ref) function:

```@docs
RxInfer.reset_session!
```

## Session Statistics

RxInfer maintains detailed statistics for each label in a session. Currently, only the `:inference` label is actively used, which collects information about inference invocations.

### What's being collected

For the `:inference` label, each invocation records:
- **Basic Information**:
  - Unique identifier (UUID)
  - Status (`:success` or `:error`)
  - Execution start and end timestamps
- **Model Information**:
  - Model source code
  - Model name
  - Inference parameters (e.g. number of iterations, free energy)
- **Data Information**:
  - Input variable names and types
  - Data characteristics
- **Error Information** (if any):
  - Error message and type

!!! note
    No actual data is collected for the `:inference` label. Only metadata such as size and type is recorded.

#### An example of a last infer call in the session

The documentation build for `RxInfer` executes real code and maintains its own session. Let's look at an example of a last infer call in the session:

```@example docs-build-stats
using RxInfer

session = RxInfer.default_session()
```

```@example docs-build-stats
stats = RxInfer.get_session_stats(session, :inference)
```

```@example docs-build-stats
last_invoke = stats.invokes[end]
```

```@example docs-build-stats
last_invoke.context
```

### Aggregated statistics

These individual invocations are then aggregated into real-time statistics:
- Total number of invocations and success/failure counts
- Success rate (fraction of successful invokes)
- Execution timing statistics (min, max, total duration)
- Set of all context keys used across invocations
- Fixed-size history of recent invocations (controlled by `DEFAULT_SESSION_STATS_CAPACITY`)

```@docs
RxInfer.SessionStats
```

You can access these statistics using `get_session_stats`:

```@example stats-example
using RxInfer

session = RxInfer.default_session()
stats = RxInfer.get_session_stats(session, :inference)

println("Total invokes: $(stats.total_invokes)")
println("Success rate: $(round(stats.success_rate * 100, digits=1))%")
println("Failed invokes: $(stats.failed_count)")
println("Mean duration (ms): $(stats.total_invokes > 0 ? round(stats.total_duration_ms / stats.total_invokes, digits=2) : 0.0)")
```

## Session Structure

A session consists of the following components:

### Session Fields
- `id::UUID`: Unique identifier for the session
- `created_at::DateTime`: Session creation timestamp
- `environment::Dict{Symbol, Any}`: System and environment information
- `stats::Dict{Symbol, SessionStats}`: Statistics per label
- `semaphore::Base.Semaphore`: Thread-safe semaphore for concurrent updates

### Environment Information
The session automatically captures system information including:
- Julia version
- RxInfer version
- Operating system
- Machine architecture
- CPU threads
- System word size

### Statistics Information
Each label's statistics (`SessionStats`) captures:
- `label::Symbol`: The label these statistics are for
- `total_invokes::Int`: Total number of invokes
- `success_count::Int`: Number of successful invokes
- `failed_count::Int`: Number of failed invokes
- `success_rate::Float64`: Fraction of successful invokes
- `min_duration_ms::Float64`: Minimum execution duration
- `max_duration_ms::Float64`: Maximum execution duration
- `total_duration_ms::Float64`: Total execution duration
- `context_keys::Set{Symbol}`: Set of all context keys used
- `invokes::CircularBuffer{SessionInvoke}`: Recent invocations history

```@docs
RxInfer.SessionInvoke
RxInfer.create_invoke
```

## Accessing Session Data

You can inspect session data to analyze inference behavior:

```@example inspect-session
using RxInfer

@model function simple_model(y)
    x ~ Normal(mean = 0.0, var = 1.0)
    y ~ Normal(mean = x, var = 1.0)
end

session = RxInfer.create_session()
result = infer(model = simple_model(), data = (y = 1.0,), session = session)

# Get inference statistics
stats = RxInfer.get_session_stats(session, :inference)

# Get the latest invoke
latest_invoke = stats.invokes[end]

# Check invocation status
println("Status: $(latest_invoke.status)")

# Calculate execution duration
duration = latest_invoke.execution_end - latest_invoke.execution_start
println("Duration: $duration")

# Access model source
println("Model name: $(latest_invoke.context[:model_name])")
println("Model source: $(latest_invoke.context[:model])")

# Examine data properties
for entry in latest_invoke.context[:data]
    println("\nData variable: $(entry.name)")
    println("Type: $(entry.type)")
    println("Size: $(entry.size)")
end
```

## [Configuration](@id session-configuration)

### Default Session

The default session is created automatically when RxInfer is first imported. It is used for logging all inference invocations by default.

```@docs
RxInfer.default_session
```

### Enabling/Disabling Logging

Session logging can be enabled or disabled globally

```@docs
RxInfer.disable_session_logging!
RxInfer.enable_session_logging!
```

### Managing Sessions

```@docs
RxInfer.create_session
RxInfer.set_default_session!
```

## Best Practices

**Error Handling**: Session logging automatically captures errors, making it easier to debug issues:
```@example error-handling
using RxInfer

@model function problematic_model(y)
    x ~ Normal(mean = 0.0, var = sqrt(-1.0)) # Invalid variance
    y ~ Normal(mean = x, var = 1.0)
end

try
    result = infer(model = problematic_model(), data = (y = 1.0,))
catch e
    # Check the latest invoke for error details
    stats = RxInfer.get_session_stats(RxInfer.default_session(), :inference)
    latest_invoke = stats.invokes[end]
    println("Status: $(latest_invoke.status)")
    println("Error: $(latest_invoke.context[:error])")
end
```

**Performance Monitoring**: Use session statistics to monitor inference performance:
```@example performance
using RxInfer, Statistics

@model function simple_model(y)
    x ~ Normal(mean = 0.0, var = 1.0)
    y ~ Normal(mean = x, var = 1.0)
end

session = RxInfer.create_session()

# Run multiple inferences
for i in 1:5
    infer(model = simple_model(), data = (y = randn(),), session = session)
end

stats = RxInfer.get_session_stats(session, :inference)
println("Mean duration (ms): $(round(stats.total_duration_ms / stats.total_invokes, digits=2))")
println("Success rate: $(round(stats.success_rate * 100, digits=1))%")
```

!!! note
    The first invocation is typically slower due to Julia's JIT compilation.

**Data Validation**: Session logging helps track data characteristics:
```@example validation
using RxInfer

@model function simple_model(y)
    x ~ Normal(mean = 0.0, var = 1.0)
    y ~ Normal(mean = x, var = 1.0)
end

session = RxInfer.create_session()
result = infer(model = simple_model(), data = (y = 1.0,), session = session)

stats = RxInfer.get_session_stats(session, :inference)
latest_invoke = stats.invokes[end]

# Check data properties
for entry in latest_invoke.context[:data]
    println("Variable: $(entry.name)")
    println("Type: $(entry.type)")
end
```

## Session Summary

You can view a tabular summary of these statistics at any time to understand how your inference tasks are performing:

!!! note
    Session statistics below are collected during the documentation build.

The main function for viewing session statistics is `summarize_session`:

```@docs
RxInfer.summarize_session
```

```@example session-stats
using RxInfer #hide
RxInfer.summarize_session(; n_last = 25)
```

The summary includes:
- Total number of inference invocations and success rate
- Execution time statistics (mean, min, max)
- List of context keys present
- Number of unique models used
- Table of most recent invocations showing:
  - Status (success/failed)
  - Duration in milliseconds
  - Model name
  - Data variables used

## Programmatic Access

If you need to access the statistics programmatically, use `get_session_stats`:

```@docs
RxInfer.get_session_stats
```

```@example session-stats
using RxInfer #hide
session = RxInfer.default_session()
stats = RxInfer.get_session_stats(session, :inference)
```

## Benchmarking Considerations

When benchmarking code that involves the `infer` function, it's important to be aware of session logging behavior:

### Why Disable Session Logging During Benchmarking?

1. **Multiple Executions**: Benchmarking tools like `BenchmarkTools.jl` execute the code multiple times to gather accurate performance metrics. Each execution is logged as a separate invoke in the session, which can quickly fill up the session buffer.

2. **Session Pollution**: These benchmark runs can pollute your session history with test invocations, making it harder to track and analyze your actual inference calls.

3. **Performance Impact**: While minimal, session logging does add some overhead to each `infer` call, which could affect benchmark results.

### Best Practices

To get accurate benchmarking results and maintain a clean session history:

```julia
# DON'T: This will fill your session with benchmark invocations
@benchmark infer(model = my_model, data = my_data)

# DO: Explicitly disable session logging during benchmarking
@benchmark infer(model = my_model, data = my_data, session = nothing)
```

You can also temporarily disable session logging globally:

```julia
# Disable session logging
previous_session = RxInfer.default_session()
RxInfer.set_default_session!(nothing)
# Run your benchmarks
# ...
# Re-enable session logging reusing the previous session
RxInfer.set_default_session!(previous_session)
```

or disable it explicitly:

```julia
RxInfer.disable_session_logging!() # works after Julia restart
```

# Developers Reference 

```@docs
RxInfer.Session
RxInfer.with_session
RxInfer.append_invoke_context
RxInfer.update_session!
RxInfer.update_stats!
```