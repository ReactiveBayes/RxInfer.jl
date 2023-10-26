# [Meta Specification](@id user-guide-meta-specification)

Some nodes in the `ReactiveMP.jl` inference engine require a meta structure that may be used to provide additional information to the nodes, or to customise the inference procedure or the way the nodes compute outbound messages. For instance, the `AR` node, modeling the Auto-Regressive process, necessitates knowledge of the order of the AR process, or the `GCV` node ([Gaussian Controlled Variance](https://ieeexplore.ieee.org/document/9173980)) needs an approximation method to handle non-conjugate relationships between variables in this node. To facilitate these requirements, `RxInfer.jl` exports `@meta` macro to specify node-specific meta and contextual information.

```@example manual_meta
using RxInfer
```

## General syntax 

`@meta` macro accepts either regular Julia `function` or a single `begin ... end` block:

```julia
@meta function MyModelMeta(arg1, arg2)
    Node(y,x) -> MetaObject(arg1, arg2)
end

my_meta = @meta begin 
    Node(y,x) -> MetaObject(arg1, arg2)
end
```

In the first case it returns a function that returns meta upon calling. For example, the meta for the `AR` node of order $5$ can be specified as follows:

```julia
@meta function ARmodel_meta(num_order)
    AR() -> ARMeta(Univariate, num_order, ARsafe())
end

my_meta = ARmodel_meta(5)
```
 
In the second case it returns constraints directly. For example, the same meta for the `AR` node can also be defined as follows:

```julia
num_order = 5

my_meta = @meta begin 
    AR() -> ARMeta(Univariate, num_order, ARsafe())
end
```
Both syntax variations provide the same meta specification and there is no preference given to one over the other. 

## Options specification 

`@meta` macro accepts optional list of options as a first argument and specified as an array of `key = value` pairs, e.g. 

```julia
my_meta = @meta [ warn = false ] begin 
   ...
end

@meta [ warn = false ] function MyModelMeta()
    ...
end
```

List of available options:
- `warn::Bool` - enables/disables various warnings with an incompatible model/meta specification

## Meta specification

First, let's start with an example:

```julia
my_meta = @meta begin 
    GCV(x, k, w) -> GCVMetadata(GaussHermiteCubature(20))
end
```

This meta specification indicates that for every `GCV` node in the model with `x`, `k` and `w` as connected variables should use the `GCVMetadata(GaussHermiteCubature(20))` meta object.

You can have a list of as many meta specification entries as possible for different nodes:

```julia
my_meta = @meta begin 
    GCV(x1, k1, w1) -> GCVMetadata(GaussHermiteCubature(20))
    AR() -> ARMeta(Univariate, 5, ARsafe())
    MyCustomNode() -> MyCustomMetaObject(arg1, arg2)
end
```

To create a model with extra constraints the user may pass an optional `meta` keyword argument for the `create_model` function:

```julia
@model function my_model(arguments...)
   ...
end

my_meta = @meta begin 
    ...
end

model, returnval = create_model(my_model(arguments...); meta = my_meta)
```

Alternatively, it is possible to use constraints directly in the automatic [`inference`](@ref) and [`rxinference`](@ref) functions that accepts `meta` keyword argument:

```julia
inferred_result = inference(
    model = my_model(arguments...),
    meta = my_meta,
    ...
)

inferred_result = rxinference(
    model = my_model(arguments...),
    meta = my_meta,
    ...
)
```

**Note**: You can also specify metadata for your nodes directly inside `@model`, without the need to use `@meta`. For example:
```julia
@model function my_model()
    ...

    y ~ AR(x, θ, γ) where { meta = ARMeta(Univariate, 5, ARsafe()) }

    ...
end
```
If you add node-specific meta to your model this way, then you do not need to use the `meta` keyword argument in the `inference` and `rxinference` functions.


## Create your own meta
Although some nodes in `RxInfer.jl` already come with their own meta structure, you have the flexibility to define different meta structures for those nodes and also for your custorm ones. A meta structure is created by using the `struct` statement in `Julia`. For example, the following snippet of code illustrates how you can create your own meta structures for your custom node and for the `AR` node:

```julia
# create your own meta structure for your custom node
struct MyCustomMeta
    arg1 
    arg2 
end

# apply the new meta structure to your node
@meta function model_meta(arg1, arg2)
    MyCustomNode() -> MyCustomMeta(arg1,arg2)
end
#---------------------------------------------------
# create your own meta structure for the AR node
struct MyARMeta
    arg1
    arg2
end

# apply the new meta structure to the AR node
@meta function model_meta(arg1, arg2)
    AR() -> MyARMeta(arg1, arg2)
end
```

## Nodes with meta data in RxInfer


|    Node                                 |   Meta                                                                   |
| :-------------------------------------- | :----------------------------------------------------------------------- |
| AR                                      | ARmeta                                                                   |
| GCV                                     | GCVMetadata                                                              |
| [DeltaFn](@id delta-node-manual)        | DeltaMeta                                                                |
| Probit                                  | Union{Nothing, ProbitMeta}                                               |
| BIFM                                    | BIFMMeta                                                                 |
| Flow                                    | FlowMeta                                                                 |
| dot                                     | Union{Nothing, MatrixCorrectionTools.AbstractCorrectionStrategy}         |
| (*)                                     | Union{Nothing, MatrixCorrectionTools.AbstractCorrectionStrategy}         |


**Notes**: The meta `Union{Nothing, ...}` in some nodes means it is optional to specify meta for those nodes.
