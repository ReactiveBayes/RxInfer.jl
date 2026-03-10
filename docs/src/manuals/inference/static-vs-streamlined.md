# [Static vs. Streaming vs. Batched Inference](@id manual-static-vs-streamlined-inference)

This guide explains the key differences between the three main inference approaches in `RxInfer`: **Static Inference**, **Streaming (Online) Inference**, and **Batched Inference**. We'll explore when to use each approach, their pros and cons, and demonstrate how to convert a model between these approaches using a Linear Gaussian State Space Model (LGSSM) as an example.

Also read about [Static Inference](@ref manual-static-inference) and [Streaming Inference](@ref manual-online-inference) for detailed information about each approach.

## [Overview of Inference Approaches](@id manual-static-vs-streamlined-overview)

### [Static Inference](@id manual-static-vs-streamlined-static)

**Static inference** processes a complete dataset all at once. It's the traditional approach where you have all your data available upfront and want to compute posterior distributions for all latent variables simultaneously.

**Key characteristics:**
- Processes entire dataset in one go
- Returns `InferenceResult` with final posteriors
- Best for offline analysis and batch processing
- Memory usage scales with dataset size
- No real-time updates

**When to use:**
- You have a complete dataset
- Offline analysis and exploration
- Batch processing workflows
- When you need all posteriors at once

### [Streaming (Online) Inference](@id manual-static-vs-streamlined-streaming)

**Streaming inference** processes data points one at a time as they arrive, continuously updating beliefs. It's designed for real-time applications where data comes in sequentially.

**Key characteristics:**
- Processes one observation at a time
- Returns `RxInferenceEngine` with reactive streams
- Real-time belief updates
- Memory usage can be controlled with history buffers
- Supports autoupdates for dynamic priors

**When to use:**
- Real-time data streams
- Online learning scenarios
- When you need immediate updates
- Interactive applications

### [Batched Inference](@id manual-static-vs-streamlined-batched)

**Batched inference** is a hybrid approach that processes data in chunks or batches, offering a middle ground between static and streaming approaches.

**Key characteristics:**
- Processes data in configurable batch sizes
- Balances memory usage and update frequency
- Can be more efficient than streaming for large datasets
- Allows for batch-level autoupdates

**When to use:**
- Large datasets that don't fit in memory
- When you want regular but not real-time updates
- Batch processing with some streaming benefits
- Memory-constrained environments

## [Comparison Table](@id manual-static-vs-streamlined-comparison)

| Aspect | Static | Streaming | Batched |
|--------|--------|-----------|---------|
| **Data Processing** | All at once | One at a time | In chunks |
| **Memory Usage** | High (scales with dataset) | Low (controlled) | Medium (configurable) |
| **Update Frequency** | Once (final result) | Real-time | Batch-level |
| **Latency** | High (wait for all data) | Low (immediate) | Medium (batch-dependent) |
| **Use Case** | Offline analysis | Real-time systems | Hybrid scenarios |
| **Return Type** | `InferenceResult` | `RxInferenceEngine` | `RxInferenceEngine` |
| **Autoupdates** | No | Yes | Yes |
| **History Tracking** | No | Yes | Yes |

## [Practical Example: Linear Gaussian State Space Model](@id manual-static-vs-streamlined-example)

Let's demonstrate the differences between these approaches using a Linear Gaussian State Space Model (LGSSM). This model is commonly used in time series analysis, signal processing, and robotics.

### [Model Definition](@id manual-static-vs-streamlined-model)

Our LGSSM has the following structure:
- **State transition**: `xₜ ~ Normal(Axₜ₋₁, Q)`
- **Observation**: `yₜ ~ Normal(Cxₜ, R)`
- **Initial state**: `x₀ ~ Normal(μ₀, Σ₀)`

Where:
- `xₜ` is the hidden state at time `t`
- `yₜ` is the observation at time `t`
- `A` is the state transition matrix
- `Q` is the state noise covariance
- `C` is the observation matrix
- `R` is the observation noise covariance

```@example manual-static-vs-streamlined
using RxInfer, Distributions, Plots, StableRNGs

# Set random seed for reproducibility
rng = StableRNG(42)

# Model parameters
n_states = 2
n_obs = 1
T = 256  # Number of time steps

# State transition matrix (simple random walk + trend)
A = [1.0 1.0; 0.0 0.9]
# State noise covariance
Q = [0.1 0.0; 0.0 0.1]
# Observation matrix
C = [1.0, 0.0]
# Observation noise variance
R = 5.0
# Initial state distribution
μ₀ = [0.0, 0.0]
Σ₀ = [1.0 0.0; 0.0 1.0]

# Generate synthetic data
function generate_lgssm_data(rng, A, Q, C, R, μ₀, Σ₀, T)
    x = Vector{Float64}[]
    y = Float64[]

    # Initial state
    push!(x, rand(rng, MvNormal(μ₀, Σ₀)))

    # Generate states and observations
    for t in 1:T
        if t > 1
            push!(x, A * x[end] + rand(rng, MvNormal(zeros(n_states), Q)))
        end
        push!(y, dot(C, x[end]) + rand(rng, NormalMeanVariance(0, R)))
    end

    return x, y
end

# Generate data
true_states, observations = generate_lgssm_data(rng, A, Q, C, R, μ₀, Σ₀, T)

p = plot(getindex.(true_states, 1), label="True state 1")
scatter!(getindex.(observations, 1), label="Observations")
```

### [1. Static Inference Implementation](@id manual-static-vs-streamlined-static-impl)

First, let's implement the static version that processes all data at once:

```@example manual-static-vs-streamlined
@model function lgssm_static(y, A, Q, C, R, μ₀, Σ₀)
    # Initial state
    x₀ ~ MvNormal(mean=μ₀, cov=Σ₀)

    # State sequence
    x[1] ~ MvNormal(mean=A * x₀, cov=Q)

    # Observations
    y[1] ~ Normal(mean=dot(C, x[1]), var=R)

    # Subsequent states and observations
    for t in 2:length(y)
        x[t] ~ MvNormal(mean=A * x[t-1], cov=Q)
        y[t] ~ Normal(mean=dot(C, x[t]), var=R)
    end
end

# Run static inference
static_results = infer(
    model=lgssm_static(A=A, Q=Q, C=C, R=R, μ₀=μ₀, Σ₀=Σ₀),
    data=(y=observations,)
)

plot(getindex.(true_states, 1), label="True state 1")
scatter!(observations, label="Observations")
plot!(getindex.(mean.(static_results.posteriors[:x]), 1), ribbon=3 .* getindex.(std.(static_results.posteriors[:x]), 1, 1), label="Static inference")
```

### [2. Streaming Inference Implementation](@id manual-static-vs-streamlined-streaming-impl)

Now let's convert this to a streaming version that processes observations one at a time:

```@example manual-static-vs-streamlined
@model function lgssm_streaming(y, x_prev, A, Q, C, R)
    # State transition from previous state
    x ~ MvNormal(mean=A * x_prev, cov=Q)

    # Observation
    y ~ Normal(mean=dot(C, x), var=R)
end

# Autoupdates for the previous state
lgssm_autoupdates = @autoupdates begin
    x_prev = mean(q(x))
end

# Initialization
lgssm_init = @initialization begin
    q(x) = MvNormal(mean = μ₀, cov = Σ₀)
end

# Run streaming inference
streaming_engine = infer(
    model=lgssm_streaming(A=A, Q=Q, C=C, R=R),
    data=(y=observations,),
    autoupdates=lgssm_autoupdates,
    initialization=lgssm_init,
    keephistory=T,
    autostart=true
)

plot(getindex.(true_states, 1), label="True state 1")
scatter!(observations, label="Observations")
plot!(getindex.(mean.(streaming_engine.history[:x]), 1), ribbon=3 .* getindex.(std.(streaming_engine.history[:x]), 1, 1), label="Streaming inference")
```

### [3. Batched Inference Implementation](@id manual-static-vs-streamlined-batched-impl)

Finally, let's implement a batched version that processes data in chunks:

```@example manual-static-vs-streamlined
# Define batch size
batch_size = 2
n_batches = ceil(Int, T / batch_size)

# Create batched datastream
function create_batched_observations(data, batch_size)
    batches = []
    for i in 1:batch_size:length(data)
        end_idx = min(i + batch_size - 1, length(data))
        push!(batches, data[i:end_idx])
    end
    return batches
end

batched_observations = create_batched_observations(observations, batch_size)

# Batched model (processes a batch of observations)
@model function lgssm_batched(y, x_prev, A, Q, C, R, batch_size)
    # First state in batch
    x[1] ~ MvNormal(mean=A * x_prev, cov=Q)
    y[1] ~ Normal(mean=dot(C, x[1]), var=R)

    # Subsequent states in batch
    for t in 2:batch_size
        x[t] ~ MvNormal(mean=A * x[t-1], cov=Q)
        y[t] ~ Normal(mean=dot(C, x[t]), var=R)
    end
end

# Autoupdates for batched processing
lgssm_batched_autoupdates = @autoupdates begin
    x_prev = mean(q(x[batch_size]))  # Use last state from previous batch
end

# Run batched inference
println("Running batched inference...")
batched_engine = infer(
    model=lgssm_batched(A=A, Q=Q, C=C, R=R, batch_size=batch_size),
    data=(y=batched_observations,),
    autoupdates=lgssm_batched_autoupdates,
    initialization=lgssm_init,
    returnvars=(:x,),
    keephistory=n_batches,
    autostart=true,
    free_energy=true
)


merged_history = vcat(batched_engine.history[:x]...)

plot(getindex.(true_states, 1), label="True state 1")
scatter!(observations, label="Observations")
plot!(getindex.(mean.(merged_history), 1), ribbon=3 .* getindex.(std.(merged_history), 1, 1), label="Batched inference")
```

## [Summary](@id manual-static-vs-streamlined-summary)

The choice between static, streaming, and batched inference depends on your specific requirements:

- **Static inference** is best for offline analysis with complete datasets
- **Streaming inference** excels at real-time applications with continuous data
- **Batched inference** provides a middle ground for large datasets with memory constraints

The Linear Gaussian State Space Model example demonstrates how the same underlying model can be adapted to different inference paradigms. All three approaches should produce different results and they differ significantly in their computational characteristics and use cases.

Remember that you can always start with static inference to validate your model and then convert to streaming or batched approaches based on your deployment requirements.