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

If a specialized node is not available, you can either [create one yourself](@ref create-node) (see also [Understanding Rules](@ref what-is-a-rule) for background on rules) or choose an appropriate approximation method for the deterministic node. For example, if all inputs to the non-linear transformation are known to be Gaussian, the fastest approximation method is probably `Linearization`. However, it requires the function to be differentiable and "nice" enough. More computationally expensive methods, such as `Unscented` or `CVIProjection`, are more robust and can be used in more general cases. We also suggest you to check [Fusing deterministic transformations with stochastic nodes](@ref inference-undefinedrules-fusedelta) example that provides additional tricks.

### Smoothing vs. Filtering

It might be appropriate to convert your model from operating on the whole dataset (smoothing) to operating on one observation at a time (filtering). Read more about smoothing in the [Static Inference](@ref manual-static-inference) section and about filtering in the [Online Inference](@ref manual-online-inference) section. It is also possible to combine both approaches and process data in batches.

## Inference Procedure Optimization

The [`infer`](@ref) function is the main entry point for inference in RxInfer.jl. It is a wrapper around the inference procedure and allows you to specify the inference algorithm, the number of iterations, the initial values for the parameters, and more. The default parameters are chosen to be a good compromise between speed and accuracy. However, in some situations, it is possible to improve the performance of the inference procedure by tuning the parameters.

### Use `free_energy = Float64` Instead of `free_energy = true`

By default, when computing free energy values, they are stored as an abstract type `Real` and are converted to `Float64` only when they are returned. This can be a significant overhead (read Julia's [Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/#Avoid-unnecessary-type-conversions)), especially for large models. The reason for this choice is that in this case, the inference procedure can be auto-differentiated where free energy values serve as the objective function. If you do not plan to auto-differentiate the inference procedure, you can set `free_energy = Float64` to avoid the overhead of type conversions.

### Be Aware of the Computational Overhead of the `limit_stack_depth` Option

RxInfer provides a `limit_stack_depth` option to limit the depth of the stack of the inference procedure, which is explained in the [Stack Overflow during inference](@ref stack-overflow-inference) section. This can be useful to avoid stack overflows, but it can also significantly degrade the performance of the inference procedure. The larger the value, the less the performance is degraded. You can tune the value based on the size of your model as well as your computer. The optimal value differs for different models and computers.

## Multicore execution

Start Julia with multiple default-pool threads, for example `julia --threads=auto`,
and pass an execution policy to the existing `infer` call:

```julia
result = infer(
    model = my_model(),
    data = my_data,
    iterations = 30,
    options = (runner = MulticoreRunner(), limit_stack_depth = 100),
)
```

`MulticoreRunner(workers = 4, min_batch_size = 32, min_work_ns = 200_000)` limits
worker tasks and avoids spawning tasks for narrow or cheap waves. A small sample
of each job type estimates numerical cost; these are actual updates, not extra
rule calls, and the first call is excluded from the timing. Set `min_work_ns = 0`
to disable cost calibration. It operates within a single
factor graph and works with the same model, constraints, initialization,
missing observations, and batch or streaming API. Each inference gets its own
execution state; the configuration can be reused across calls. Model construction
still runs serially.

This is an experimental **wave-based message schedule**. The runner captures
ready inputs on the inference task, evaluates independent message rules,
variable marginal products, and joint marginal rules in parallel, then delivers
results in a deterministic order. Equality-chain caches and reactive subscribers
are updated only on the inference task. Worker failures are joined and propagated
before publishing the failed wave; `catch_exception = true` works as usual in
batch inference.

One and multiple workers use the same wave schedule. The standard runner uses a
depth-first schedule, so intermediate beliefs, free energies, and the number of
iterations to convergence can differ. Compare converged results and time to the
same accuracy when evaluating speedup. A different schedule can also affect
convergence in nonlinear or loopy models; check the convergence of your model.

Rules accessing a factor-node object, mutable metadata, callbacks, or annotation
processors use serial computation. This protects shared RNGs and approximation
buffers. Custom rules that are eligible for parallel execution must not mutate
their input distributions or shared global state. Existing custom nodes remain
usable through the serial fallback. A custom per-node stream postprocessor can
also keep that node on its existing execution path. The `runner` option cannot
be combined with a global `stream_postprocessors` option; `limit_stack_depth`
is supported directly.

Speedup depends on the amount of independent numerical work. Large collections
of expensive rules can benefit, while cheap scalar rules, narrow chains,
model construction, and memory bandwidth can dominate other models. More nodes
alone do not guarantee speedup. Benchmark warmed runs with `KeepLast()` and
without detailed tracing. If the rules call BLAS, compare with
`LinearAlgebra.BLAS.set_num_threads(1)` to avoid nested thread oversubscription;
the runner does not alter global BLAS settings.

Reproducible benchmarks live in `benchmarks/multicore_scaling.jl`,
`benchmarks/multicore_grid.jl`, and `benchmarks/multicore_rslds.jl`. They check
posterior agreement across worker counts and report warmed wall-clock timings.

## Experimental compact backend

`CompiledRunner` is a separate, opt-in backend for a single model. It constructs
a compact graph and executes existing message, marginal and product rules through
an indexed schedule, without reactive subjects or subscriptions at each factor.
It is not batching and does not replace a grid with a specialized linear solver.
The default backend and `MulticoreRunner` are unchanged.

```julia
result = infer(
    model = my_model(),
    data = my_data,
    initialization = my_initialization,
    iterations = 200,
    returnvars = KeepLast(),
    options = (runner = CompiledRunner(workers = 6),),
)
```

Start Julia with at least as many threads as requested workers. The runner does
not change BLAS settings. `CompiledRunner(workers = 1)` uses the same compiled
schedule without parallel worker tasks. Batch and streaming/autoupdate entry
points are supported; streaming uses Rocket only at the public input/output
boundary. Each inference instance owns its execution state.

This backend currently requires the modified local RxInfer, ReactiveMP and
GraphPPL packages. The reproducible environment, commands, supported examples
and validation record are in `benchmarks/compiled/README.md` and
`benchmarks/COMPILED_BACKEND_PROGRESS.md`; registry GraphPPL alone is insufficient.

Compatibility is experimental, not established for every model. Conjugate
models, structured HMMs, several mixture layouts, nonlinear Delta/CVI models,
predictions and streaming have regression fixtures. The RSLDS notebook's custom
Gate node has an explicit lowering adapter. Other imperative custom node layouts
may also need adapters. Unsupported layouts, stream postprocessors and node
contraction raise explicit errors; there is no silent reactive fallback.
Stateful or unaudited rule kernels execute serially within the compiled backend.
Failed inference state cannot be reused; create a fresh inference instance.

Compare converged means and variances, not iteration-by-iteration trajectories:
the schedule can require a different number of sweeps and may affect convergence
on loopy or nonlinear models. Retaining `KeepEach()` histories can dominate
memory, and larger models do not guarantee linear core scaling. The compact
backend also has compilation/setup costs that may outweigh inference savings
on small models. Measure time to equal accuracy with BenchmarkTools, including
construction and result materialization; see `benchmarks/compiled_benchmark_tools.jl`.

## Getting Help

If you encounter performance issues:

1. **Check the documentation**: Review relevant sections for optimization tips
2. **Use the community**: Open discussions on GitHub for specific issues
3. **Profile your code**: Use Julia's profiling tools to identify bottlenecks
4. **Start simple**: Build complexity gradually to identify performance issues
