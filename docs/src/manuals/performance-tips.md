# [Performance Tips](@id user-guide-performance-tips)

This section provides practical advice and best practices for optimizing the performance of your RxInfer models. Following these guidelines can significantly improve inference speed and memory efficiency.

!!! note 
    Before diving into RxInfer-specific optimizations, we strongly recommend reading Julia's official [Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/) guide. Many performance improvements come from following Julia's general best practices, such as avoiding global variables, using type stability, and minimizing allocations. The tips in this section build upon those fundamental principles.

## Julia Compilation Latency

Julia uses **Just-In-Time (JIT)** compilation. The **first time** you run a model and inference procedure, Julia compiles the specialized machine code. This can cause noticeable delays **only once**. Afterward, execution becomes much faster. This might be especially problematic for models and factor nodes that accept a dynamic number of arguments. Such nodes include mixture nodes (where the number of components is only known at compilation time) as well as deterministic nodes representing non-linear transformations (since those transformations can be arbitrary, their signature is only known at compilation time).

**Tips:** 
- Don't worry about long first-run times during development — focus on steady-state performance.
- Use the `@time` macro from Julia to investigate the time spent on compilation and execution.

## Model Structure Optimization

RxInfer is designed for fast inference on factor graphs and leverages the model structure to optimize the inference procedure. However, it is always possible to create a huge model with complex dependencies between variables and make inference slow with RxInfer. 

**General guidelines for model structure optimization:**

### Choose Appropriate Parametrization for Your Nodes

While confusing at first glance, the choice of parametrization for your nodes can have a significant impact on the performance of the inference procedure. For this reason, RxInfer allows you to choose, for example, between `NormalMeanPrecision` and `NormalMeanVariance` parametrizations for `Normal` nodes. Or, you can choose between `MvNormalMeanPrecision` and `MvNormalMeanScalePrecision` parametrizations for `MvNormal` nodes. The difference between these parametrizations is that the former needs to store the entire precision matrix, while the latter uses a single number to store the scale of the diagonal of the precision matrix.

### Use Conjugate Pairs

[Conjugate pairs](https://en.wikipedia.org/wiki/Conjugate_prior) enable analytical message updates. For example, a `Gamma` prior is appropriate for a `NormalMeanPrecision` node, but an `InverseGamma` is not. Conversely, an `InverseGamma` prior is appropriate for a `NormalMeanVariance` node, but a `Gamma` is not. Another example is that a `Beta` prior is appropriate for a `Bernoulli` node, but a `Binomial` is not. A `Wishart` prior is appropriate for an `MvNormalMeanPrecision` node, and an `InverseWishart` is appropriate for an `MvNormalMeanCovariance` node.  Note that the conjugacy also depends on the local factorization of your model. If you place priors on both the mean and the precision in `MvNormalMeanPrecision`, you must enforce independence (e.g., `q(μ,Λ)=q(μ)q(Λ)`) to make the model conditionally conjugate.
### Be Aware of the Computational Overhead of Deterministic Nodes

Each deterministic node adds computational overhead and requires approximation method specification. Read more about approximation methods in the [Deterministic nodes](@ref delta-node-manual) section. In some situations, however, it is possible to use specialized factor nodes instead of deterministic nodes. For example, the `SoftDot` node is a specialized factor node for computing the dot product of two vectors where the result is passed to a `Normal` node. Using `SoftDot` directly instead of `Normal(mean = dot(...), ...)` can significantly improve both the performance and accuracy of the inference procedure. Similar applies to `ContinuousTransition` node.

If a specialized node is not available, you can either [create one yourself](@ref create-node) or choose an appropriate approximation method for the deterministic node. For example, if all inputs to the non-linear transformation are known to be Gaussian, the fastest approximation method is probably `Linearization`. However, it requires the function to be differentiable and "nice" enough. More computationally expensive methods, such as `Unscented` or `CVIProjection`, are more robust and can be used in more general cases. We also suggest you to check [Fusing deterministic transformations with stochastic nodes](@ref inference-undefinedrules-fusedelta) example that provides additional tricks.

### Smoothing vs. Filtering

It might be appropriate to convert your model from operating on the whole dataset (smoothing) to operating on one observation at a time (filtering). Read more about smoothing in the [Static Inference](@ref manual-static-inference) section and about filtering in the [Online Inference](@ref manual-online-inference) section. It is also possible to combine both approaches and process data in batches.

## Inference Procedure Optimization

The [`infer`](@ref) function is the main entry point for inference in RxInfer.jl. It is a wrapper around the inference procedure and allows you to specify the inference algorithm, the number of iterations, the initial values for the parameters, and more. The default parameters are chosen to be a good compromise between speed and accuracy. However, in some situations, it is possible to improve the performance of the inference procedure by tuning the parameters.

### Use `free_energy = Float64` Instead of `free_energy = true`

By default, when computing free energy values, they are stored as an abstract type `Real` and are converted to `Float64` only when they are returned. This can be a significant overhead (read Julia's [Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/#Avoid-unnecessary-type-conversions)), especially for large models. The reason for this choice is that in this case, the inference procedure can be auto-differentiated where free energy values serve as the objective function. If you do not plan to auto-differentiate the inference procedure, you can set `free_energy = Float64` to avoid the overhead of type conversions.

### Be Aware of the Computational Overhead of the `limit_stack_depth` Option

RxInfer provides a `limit_stack_depth` option to limit the depth of the stack of the inference procedure, which is explained in the [Stack Overflow during inference](@ref stack-overflow-inference) section. This can be useful to avoid stack overflows, but it can also significantly degrade the performance of the inference procedure. The larger the value, the less the performance is degraded. You can tune the value based on the size of your model as well as your computer. The optimal value differs for different models and computers.

## Getting Help

If you encounter performance issues:

1. **Check the documentation**: Review relevant sections for optimization tips
2. **Use the community**: Open discussions on GitHub for specific issues
3. **Profile your code**: Use Julia's profiling tools to identify bottlenecks
4. **Start simple**: Build complexity gradually to identify performance issues
