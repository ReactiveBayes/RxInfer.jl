# Session Logging

RxInfer provides a built-in session logging system that helps track and analyze various aspects of RxInfer usages. This feature is particularly useful for debugging, performance monitoring, and understanding the behavior of your inference models.

## Overview

Session logging in RxInfer automatically captures:
- Model source code
- Input data characteristics and metadata
- Execution timing
- Success/failure status
- Error information (if any)
- Environment information (Julia version, OS, etc.)

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

# Show the number of logged invocations
println("Number of invokes: $(length(session.invokes))")
```

!!! note
    The number of logged invocations may be different from the number of invocations in the example above
    since the session is created and logged at the start of the documentation build.

## Session Capacity

By default, RxInfer maintains a fixed-size history of the last 1000 inference invocations. 
When this limit is exceeded, the oldest invocations are automatically dropped. This prevents 
memory growth while maintaining recent history.

You can customize the capacity when creating a session:

```@example custom-session
using RxInfer

# Create a session that keeps last 100 invokes
session = RxInfer.create_session(capacity = 100)

# Create a session with larger history
large_session = RxInfer.create_session(capacity = 5000)

println("Session capacity: $(capacity(session.invokes))")
println("Large session capacity: $(capacity(large_session.invokes))")
```

This is particularly useful when:
- Running benchmarks that might generate many invocations
- Working with long-running applications
- Managing memory usage in resource-constrained environments

You can also create and use custom sessions:

```@example custom-session
using RxInfer

@model function simple_model(y)
    x ~ Normal(mean = 0.0, var = 1.0)
    y ~ Normal(mean = x, var = 1.0)
end

# Create a custom session with capacity of 10 invokes
session = RxInfer.create_session(capacity = 10)

# Run inference with custom session
result = infer(
    model = simple_model(),
    data = (y = 1.0,),
    session = session
)

println("Session ID: $(session.id)")
println("Created at: $(session.created_at)")
```

## Session Structure

A session consists of the following components:

### Session Fields
- `id::UUID`: Unique identifier for the session
- `created_at::DateTime`: Session creation timestamp
- `environment::Dict{Symbol, Any}`: System and environment information
- `invokes::CircularBuffer{SessionInvoke}`: Fixed-size circular buffer of inference invocations

### Environment Information
The session automatically captures system information including:
- Julia version
- RxInfer version
- Operating system
- Machine architecture
- CPU threads
- System word size

### Invoke Information
Each inference invocation (`SessionInvoke`) captures:
- `id::UUID`: Unique identifier for the invocation
- `status::Symbol`: Status of the invocation (`:success`, `:failed`, or `:unknown`)
- `execution_start::DateTime`: Start timestamp
- `execution_end::DateTime`: End timestamp
- `context::Dict{Symbol, Any}`: Contextual information about the invocation

## Accessing Session Data

You can inspect session data to analyze inference behavior:

```@example inspect-session
using RxInfer

@model function simple_model(y)
    x ~ Normal(mean = 0.0, var = 1.0)
    y ~ Normal(mean = x, var = 1.0)
end

session = RxInfer.create_session(capacity = 100)
result = infer(model = simple_model(), data = (y = 1.0,), session = session)

# Get the latest invoke
latest_invoke = session.invokes[end]

# Check invocation status
println("Status: $(latest_invoke.status)")

# Calculate execution duration
duration = latest_invoke.execution_end - latest_invoke.execution_start
println("Duration: $duration")

# Access model source
println("Model name: $(latest_invoke.context[:model_name])")
println("Model source: $(latest_invoke.context[:model_source])")

# Examine data properties
for entry in latest_invoke.context[:data]
    println("\nData variable: $(entry.name)")
    println("Type: $(entry.type)")
    println("Size: $(entry.size)")
end
```

## Configuration

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

1. **Error Handling**: Session logging automatically captures errors, making it easier to debug issues:
```@example error-handling
using RxInfer

@model function problematic_model(y)
    x ~ Normal(mean = 0.0, var = -1.0) # Invalid variance
    y ~ Normal(mean = x, var = 1.0)
end

try
    result = infer(model = problematic_model(), data = (y = 1.0,))
catch e
    # Check the latest invoke for error details
    latest_invoke = RxInfer.default_session().invokes[end]
    println("Status: $(latest_invoke.status)")
    println("Error: $(latest_invoke.context[:error])")
end
```

2. **Performance Monitoring**: Use session data to monitor inference performance:
```@example performance
using RxInfer, Statistics

@model function simple_model(y)
    x ~ Normal(mean = 0.0, var = 1.0)
    y ~ Normal(mean = x, var = 1.0)
end

session = RxInfer.create_session(capacity = 100)

# Run multiple inferences
for i in 1:5
    infer(model = simple_model(), data = (y = randn(),), session = session)
end

durations = map(session.invokes) do invoke
    invoke.execution_end - invoke.execution_start
end
```

3. **Data Validation**: Session logging helps track data characteristics:
```@example validation
using RxInfer

@model function simple_model(y)
    x ~ Normal(mean = 0.0, var = 1.0)
    y ~ Normal(mean = x, var = 1.0)
end

session = RxInfer.create_session(capacity = 100)
result = infer(model = simple_model(), data = (y = 1.0,), session = session)

for entry in session.invokes[end].context[:data]
    println("Variable '$(entry.name)' size: $(entry.size)")
end
```

## Session Statistics

RxInfer automatically collects statistics about inference runs. You can view these statistics at any time to understand how your inference tasks are performing.

!!! note
    Session statistics below are collected during the documentation build.

## Viewing Statistics

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
RxInfer.get_session_stats()
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

# Developers reference 


```@docs
RxInfer.Session
RxInfer.with_session
RxInfer.create_invoke
RxInfer.append_invoke_context
```