# [Benchmark callbacks](@id manual-inference-benchmark-callbacks)

```@meta
CurrentModule = RxInfer
```

`RxInfer` provides a built-in callback structure called [`RxInferBenchmarkCallbacks`](@ref) for collecting timing information during the inference procedure.
This structure aggregates timestamps across multiple inference runs, allowing you to track performance statistics (min/max/average/etc.) of your model's creation and inference procedure.
For general information about the callbacks system, see [Callbacks](@ref manual-inference-callbacks).

## Basic usage

```@example manual-inference-benchmark-callbacks
using RxInfer
using Test #hide

@model function iid_normal(y)
    μ  ~ Normal(mean = 0.0, variance = 100.0)
    γ  ~ Gamma(shape = 1.0, rate = 1.0)
    y .~ Normal(mean = μ, precision = γ)
end

init = @initialization begin
    q(μ) = vague(NormalMeanVariance)
end

# Warm up to avoid measuring compilation time
infer(model = iid_normal(), data = (y = randn(100),), constraints = MeanField(), iterations = 5, initialization = init, callbacks = RxInferBenchmarkCallbacks()) #hide

# Create a benchmark callbacks instance to track performance
benchmark_callbacks = RxInferBenchmarkCallbacks()

# Run inference multiple times to gather statistics
for i in 1:5
    infer(
        model = iid_normal(),
        data = (y = randn(100),),
        constraints = MeanField(),
        iterations = 5,
        initialization = init,
        callbacks = benchmark_callbacks,
    )
end

@test length(benchmark_callbacks.before_model_creation_ts) == 5 #hide
@test length(benchmark_callbacks.after_model_creation_ts) == 5 #hide
@test length(benchmark_callbacks.before_inference_ts) == 5 #hide
@test length(benchmark_callbacks.after_inference_ts) == 5 #hide
nothing #hide
```

The benchmark callbacks instance accumulates timestamps across multiple calls to [`infer`](@ref), making it easy to collect performance statistics over many runs.

## Displaying results

Install `PrettyTables.jl` to display the collected statistics in a nicely formatted table:

```@example manual-inference-benchmark-callbacks
using PrettyTables

PrettyTables.pretty_table(benchmark_callbacks)
```

## Accessing from model metadata

After model creation, the benchmark callbacks instance is automatically saved into the model's metadata under the `:benchmark` key. This makes it accessible from the inference result without needing to hold onto the callbacks object separately:

```@example manual-inference-benchmark-callbacks
result = infer(
    model = iid_normal(),
    data = (y = randn(100),),
    constraints = MeanField(),
    iterations = 5,
    initialization = init,
    callbacks = RxInferBenchmarkCallbacks(),
)

@test haskey(result.model.metadata, :benchmark) #hide
@test result.model.metadata[:benchmark] isa RxInferBenchmarkCallbacks #hide

benchmark = result.model.metadata[:benchmark]
println(benchmark)
```

## Tracked events

The `RxInferBenchmarkCallbacks` structure collects timestamps at the following stages:

| Event | Batch inference | Streamline inference |
|:------|:---:|:---:|
| Model creation (`before`/`after`) | yes | yes |
| Inference (`before`/`after`) | yes | — |
| Each iteration (`before`/`after`) | yes | — |
| Autostart (`before`/`after`) | — | yes |

## Buffer capacity

By default, the structure uses circular buffers with a capacity of [`RxInfer.DEFAULT_BENCHMARK_CALLBACKS_BUFFER_CAPACITY`](@ref) entries.
This limits memory usage in long-running applications. You can change the capacity:

```@example manual-inference-benchmark-callbacks
# Store up to 10000 benchmark entries
large_buffer_callbacks = RxInferBenchmarkCallbacks(capacity = 10_000)
nothing #hide
```

## Programmatic access to statistics

Use [`RxInfer.get_benchmark_stats`](@ref) to retrieve the raw statistics matrix:

```@example manual-inference-benchmark-callbacks
# Use the previously populated benchmark_callbacks
stats = RxInfer.get_benchmark_stats(benchmark_callbacks)

@test size(stats, 2) == 6 #hide
@test size(stats, 1) >= 1 #hide

for row in eachrow(stats)
    println(row[1], ": min=", round(row[2] / 1e6, digits=2), "ms, mean=", round(row[4] / 1e6, digits=2), "ms")
end
```

The matrix contains the following columns:
1. Operation name (`String`)
2. Minimum time (`Float64`, nanoseconds)
3. Maximum time (`Float64`, nanoseconds)
4. Mean time (`Float64`, nanoseconds)
5. Median time (`Float64`, nanoseconds)
6. Standard deviation (`Float64`, nanoseconds)

## API Reference

```@docs
RxInferBenchmarkCallbacks
RxInfer.get_benchmark_stats
RxInfer.DEFAULT_BENCHMARK_CALLBACKS_BUFFER_CAPACITY
```

!!! note
    The timing measurements include all overhead from the Julia runtime and may vary between runs. For more precise benchmarking of specific code sections, consider using the `BenchmarkTools.jl` package.
