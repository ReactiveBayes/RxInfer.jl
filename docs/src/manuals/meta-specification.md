# [Meta Specification](@id user-guide-meta-specification)

`RxInfer.jl` utilizes the `GraphPPL.jl` package to construct a factor graph representing a probabilistic model, and then employs the `ReactiveMP.jl` package to conduct variational inference through message passing on this factor graph. Some factor nodes within the `ReactiveMP.jl` inference engine require an additional structure, known as meta-information. This meta-information can serve various purposes such as providing extra details to nodes, customizing the inference process, or adjusting how nodes compute outgoing messages. For example, the `AR` node, which models _Auto-Regressive_ processes, needs to know the order of the `AR` process. Similarly, the `GCV` node ([Gaussian Controlled Variance](https://ieeexplore.ieee.org/document/9173980)) requires an approximation method to handle non-conjugate relationships between its variables. To address these needs, `RxInfer.jl` utilizes the `@meta` macro from the `GraphPPL.jl` package to specify node-specific meta-information and contextual details.

Here, we only touch upon the basics of the `@meta` macro. For further details, please consult the [official documentation](https://reactivebayes.github.io/GraphPPL.jl/stable/) of the GraphPPL.jl package.

## General syntax 

The `@meta` macro accepts either a regular Julia function or a single `begin ... end` block:

```@example manual_meta
using RxInfer

struct MetaObject
    arg1
    arg2
end

@meta function create_meta(arg1, arg2)
    Normal(y, x) -> MetaObject(arg1, arg2)
end

my_meta = @meta begin 
    Normal(y, x) -> MetaObject(1, 2)
end

nothing #hide
```

In the first case, it returns a function that produces an object containing metadata when called. For instance, to specify meta for an `AR` node with an order of $5$, you can do the following:

```@example manual_meta
@meta function ARmodel_meta(num_order)
    AR() -> ARMeta(Multivariate, num_order, ARsafe())
end

my_meta = ARmodel_meta(5)
nothing #hide
```
 
In the second case, it directly provides the meta object. The same meta for the `AR` node can also be defined as follows:

```@example manual_meta
num_order = 5

my_meta = @meta begin 
    AR() -> ARMeta(Multivariate, num_order, ARsafe())
end
nothing #hide
```

Both syntax variations provide the same meta specification and there is no preference given to one over the other. 

Another example:

```@example manual_meta
my_meta = @meta begin 
    GCV(x, k, w) -> GCVMetadata(GaussHermiteCubature(20))
end
nothing #hide
```

This meta specification indicates that for every `GCV` node in the model with `x`, `k` and `w` as connected variables should use the `GCVMetadata(GaussHermiteCubature(20))` meta object.

You can have a list of as many meta specification entries as possible for different nodes:

```@example manual_meta
my_meta = @meta begin 
    GCV(x1, k1, w1) -> GCVMetadata(GaussHermiteCubature(20))
    AR() -> ARMeta(Multivariate, 5, ARsafe())
end
nothing #hide
```

The meta-information object can be used in the [`infer`](@ref) function that accepts `meta` keyword argument:

```julia
inferred_result = infer(
    model = my_model(arguments...),
    data  = ...,
    meta  = my_meta,
    ...
)
```

Users can also specify metadata for nodes directly inside `@model`, without the need to use `@meta`. For example:

```julia
@model function my_model()
    ...

    y ~ AR(x, θ, γ) where { meta = ARMeta(Multivariate, 5, ARsafe()) }

    ...
end
```

If you add node-specific meta to your model this way, you do not need to use the `meta` keyword argument in the `infer` function.

## Create your own meta

Although some nodes in `RxInfer.jl` already come with their own meta structure, users have the flexibility to define different meta structures for those nodes and also for custom ones. A meta structure is created by using the `struct` statement in `Julia`. For example, the following snippet of code illustrates how you can create your own meta structures for your custom node. This section provides a concrete example of how to create and use meta in `RxInfer.jl`. Suppose that we have the following Gaussian model:

$$\begin{aligned}
 x & \sim \mathrm{Normal}(2.5, 0.5)\\
 y & \sim \mathrm{Normal}(2*x, 2.0)
\end{aligned}$$

where $y$ is observable data and $x$ is a latent variable. In `RxInfer.jl`, the inference procedure for this model is well defined without the need of specifying any meta data for the `Normal` node.

```@example custom-meta
using RxInfer

#create data
y_data = 4.0 

#make model
@model function gaussian_model(y)
    x ~ NormalMeanVariance(2.5, 0.5)
    y ~ NormalMeanVariance(2*x, 2.)
end

#do inference
inference_result = infer(
    model = gaussian_model(),
    data = (y = y_data,)
)
```

However, let's say we would like to experiment with message update rules and define a new inference procedure by introducing a meta structure to the `Normal` node that always yields a message equal to `Normal` distribution with mean $m$ clamped between `lower_limit` and `upper_limit` for the outbound messages of the node. This is done as follows:

```@example custom-meta
#create your new meta structure for Normal node
struct MetaConstrainedMeanNormal{T}
    lower_limit :: T
    upper_limit :: T
end

#define rules with meta for the Normal node
@rule NormalMeanVariance(:out, Marginalisation) (q_μ::Any, q_v::Any, meta::MetaConstrainedMeanNormal) = begin
    return NormalMeanVariance(clamp(mean(q_μ), meta.lower_limit, meta.upper_limit), mean(q_v))
end

@rule NormalMeanVariance(:μ, Marginalisation) (q_out::Any, q_v::Any, meta::MetaConstrainedMeanNormal) = begin
    return NormalMeanVariance(clamp(mean(q_out), meta.lower_limit, meta.upper_limit), mean(q_v))
end
```

```@example custom-meta
#make model
@model function gaussian_model_with_meta(y)
    x ~ NormalMeanVariance(2.5, 0.5)
    y ~ NormalMeanVariance(2*x, 2.)
end

custom_meta = @meta begin
    NormalMeanVariance(y) -> MetaConstrainedMeanNormal(-2, 2)
end

#do inference
inference_result = infer(
    model = gaussian_model(),
    data = (y = y_data,),
    meta = custom_meta
)

println("Estimated mean for latent state `x` is ", mean(inference_result.posteriors[:x]), " with standard deviation ", std(inference_result.posteriors[:x]))
```

!!! warning 
    The above example is not mathematically correct. It is only used to show how we can work with `@meta` as well as how to create a meta structure for a node in `RxInfer.jl`.

Read more about the `@meta` macro in the [official documentation](https://reactivebayes.github.io/GraphPPL.jl/stable/) of GraphPPL

## Adding metadata to nodes in submodels

Similarly to the `@constraints` macro, the `@meta` macro exposes syntax to push metadata to nodes in submodels. With the `for meta in submodel` syntax we can apply metadata to nodes in submodels. For example, if we use the `gaussian_model_with_meta` mnodel in a larger model, we can write:

```@example custom-meta
custom_meta = @meta begin
    for meta in gaussian_model_with_meta
        NormalMeanVariance(y) -> MetaConstrainedMeanNormal(-2, 2)
    end
end
```