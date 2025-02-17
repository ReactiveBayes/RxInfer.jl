# [Stack Overflow during inference](@id stack-overflow-inference)

When working with large probabilistic models in RxInfer, you might encounter a `StackOverflowError`. This section explains why this happens and how to prevent it.

## The Problem

RxInfer uses reactive streams to compute messages between nodes in the factor graph. The subscription to these streams happens recursively, which means:

1. Each node subscribes to its input messages or posteriors
2. Those input messages may need to subscribe to their own inputs
3. This continues until all dependencies are resolved

For large models, this recursive subscription process can consume the entire stack space, resulting in a `StackOverflowError`.

## Example Error

When this occurs, you'll see an error message that looks something like this:

```julia
ERROR: Stack overflow error occurred during the inference procedure. 
```

## Solution: Limiting Stack Depth

RxInfer provides a solution through the `limit_stack_depth` option in the inference options. This option limits the recursion depth at the cost of some performance overhead.

### How to Use

You can enable stack depth limiting by passing it through the `options` parameter to the `infer` function:

```@example stack-overflow-inference
using RxInfer

@model function long_state_space_model(y)
    x[1] ~ Normal(mean = 0.0, var = 1.0)
    y[1] ~ Normal(mean = x[1], var = 1.0)
    for i in 2:length(y)
        x[i] ~ Normal(mean = x[i - 1], var = 1.0)
        y[i] ~ Normal(mean = x[i], var = 1.0)
    end
end

data = (y = randn(10000), )

results = infer(
    model = long_state_space_model(),
    data = data,
    options = (
        limit_stack_depth = 100, # note the comma
    )
)
```

!!! note
    Note the comma after `limit_stack_depth = 100`. This is important because it tells Julia that the option is placed in the named tuple `options`.

Without `limit_stack_depth` enabled, the inference will fail with a `StackOverflowError`

```julia
results = infer(
    model = long_state_space_model(),
    data = data
)
```

```julia
ERROR: Stack overflow error occurred during the inference procedure. 
```

### Performance Considerations

When `limit_stack_depth` is enabled:
- The recursive subscription process is split into multiple steps
- This prevents stack overflow but introduces performance overhead (you should verify this in your use case)
- For very large models, this option might be essential for successful execution

## When to Use

Consider using `limit_stack_depth` when:
- Working with large models (many nodes/variables)
- Encountering `StackOverflowError`
- Processing deep hierarchical models
- Dealing with long sequences or time series

!!! tip
    If you're not sure whether you need this option, try running your model without it first. Only enable `limit_stack_depth` if you encounter stack overflow issues.

## Further Reading

For more details about inference options and execution, see:
- [Static Inference](@ref manual-static-inference) documentation
- The `options` parameter in the [`infer`](@ref) function documentation
