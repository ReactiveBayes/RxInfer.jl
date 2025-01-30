# Session Logging

RxInfer provides a built-in session logging system that helps track and analyze inference invocations. This feature is particularly useful for debugging, performance monitoring, and understanding the behavior of your inference models.

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

### Enabling/Disabling Logging

Session logging can be enabled or disabled globally:

```@example config-session
using RxInfer

# Disable session logging (takes effect after Julia restart)
RxInfer.disable_session_logging!()

# Enable session logging (takes effect after Julia restart)
RxInfer.enable_session_logging!()
```

You can also disable logging for specific inference calls:

```@example disable-session
using RxInfer

@model function simple_model(y)
    x ~ Normal(mean = 0.0, var = 1.0)
    y ~ Normal(mean = x, var = 1.0)
end

# Run inference without logging
result = infer(
    model = simple_model(),
    data = (y = 1.0,),
    session = nothing
)
```

### Managing Sessions

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

## Thread Safety

The session logging system is thread-safe, using a semaphore to protect access to the default session:

```@example thread-safety
using RxInfer

# Thread-safe access to default session
session = RxInfer.default_session()

# Thread-safe session update
new_session = RxInfer.create_session()
RxInfer.set_default_session!(new_session)
``` 