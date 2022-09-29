# [Meta Specification](@id user-guide-meta-specification)

Some nodes in the `ReactiveMP.jl` inference engine accept optional meta structure that may be used to change or customise the inference procedure or the way node computes outbound messages. As an example `GCV` node accepts the approximation method that will be used to approximate non-conjugate relationships between variables in this node. `RxInfer.jl` exports `@meta` macro to specify node-specific meta and contextual information. For example:

## General syntax 

`@meta` macro accepts either regular Julia function or a single `begin ... end` block. For example both are valid:

```julia

@meta function create_meta(arg1, arg2)
    ...
end

mymeta = @meta begin 
    ...
end
```

In the first case it returns a function that returns meta upon calling, e.g. 

```julia
@meta function create_meta(flag)
    ...
end

mymeta = create_meta(true)
```
 
and in the second case it returns constraints directly.

```julia
mymeta = @meta begin 
    ...
end
```

## Options specification 

`@meta` macro accepts optional list of options as a first argument and specified as an array of `key = value` pairs, e.g. 

```julia
mymeta = @meta [ warn = false ] begin 
   ...
end
```

List of available options:
- `warn::Bool` - enables/disables various warnings with an incompatible model/meta specification

## Meta specification

First, let's start with an example:

```@example manual_meta
meta = @meta begin 
    GCV(x, k, w) -> GCVMetadata(GaussHermiteCubature(20))
end
```

This meta specification indicates, that for every `GCV` node in the model with `x`, `k` and `w` as connected variables should use the `GCVMetadata(GaussHermiteCubature(20))` meta object.

You can have a list of as many meta specification entries as possible for different nodes:

```@example manual_meta
meta = @meta begin 
    GCV(x1, k1, w1) -> GCVMetadata(GaussHermiteCubature(20))
    GCV(x2, k2, w3) -> GCVMetadata(GaussHermiteCubature(30))
    NormalMeanVariance(y, x) -> MyCustomMetaObject(arg1, arg2)
end
```

To create a model with extra constraints the user may pass an optional `meta` positional argument for the model generator function:

```julia
@model function my_model(arguments...)
   ...
end

meta = @meta begin 
    ...
end

model, returnval = my_model(arguments...)(; meta = meta)
```

Alternatively, it is possible to use constraints directly in the automatic [`inference`](@ref) and [`rxinference`](@ref) functions that accepts `meta` keyword argument. 