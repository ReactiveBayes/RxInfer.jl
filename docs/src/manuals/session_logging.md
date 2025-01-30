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

You can also create and use custom sessions:

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

## Session Structure

A session consists of the following components:

### Session Fields
- `id::UUID`: Unique identifier for the session
- `created_at::DateTime`: Session creation timestamp
- `environment::Dict{Symbol, Any}`: System and environment information
- `invokes::Vector{SessionInvoke}`: List of inference invocations

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

session = RxInfer.create_session()
result = infer(model = simple_model(), data = (y = 1.0,), session = session)

# Get the latest invoke
latest_invoke = session.invokes[end]

# Check invocation status
println("Status: $(latest_invoke.status)")

# Calculate execution duration
duration = latest_invoke.execution_end - latest_invoke.execution_start
println("Duration: $duration")

# Access model source
println("Model source: $(latest_invoke.context[:model])")

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

```@example manage-session
using RxInfer

# Create a new session
new_session = RxInfer.create_session()

# Set as default session
RxInfer.set_default_session!(new_session)

# Clear default session
RxInfer.set_default_session!(nothing)
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

session = RxInfer.create_session()

# Run multiple inferences
for i in 1:5
    infer(model = simple_model(), data = (y = randn(),), session = session)
end

durations = map(session.invokes) do invoke
    invoke.execution_end - invoke.execution_start
end

println("Mean duration: $(mean(durations))")
println("Std duration: $(std(durations))")
```

3. **Data Validation**: Session logging helps track data characteristics:
```@example validation
using RxInfer

@model function simple_model(y)
    x ~ Normal(mean = 0.0, var = 1.0)
    y ~ Normal(mean = x, var = 1.0)
end

session = RxInfer.create_session()
result = infer(model = simple_model(), data = (y = [1.0, 2.0],), session = session)

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
RxInfer.summarize_session()
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

# Developers reference 


```@docs
RxInfer.Session
RxInfer.with_session
RxInfer.create_invoke
RxInfer.append_invoke_context
```