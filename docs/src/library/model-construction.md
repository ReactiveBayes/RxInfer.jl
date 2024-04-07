# [Model construction in RxInfer](@id lib-model-construction)

Model creation in `RxInfer` largely depends on [`GraphPPL`](https://github.com/ReactiveBayes/GraphPPL.jl) package.
`RxInfer` re-exports the `@model` macro from `GraphPPL` and defines extra plugins and data structures on top of the default functionality.

!!! note
    The model creation and construction were largely refactored in `GraphPPL` v4. 
    Read [_Migration Guide_](https://reactivebayes.github.io/GraphPPL.jl/stable/migration_3_to_4/) for more details.

Also read the [_Model Specification_](@ref user-guide-model-specification) guide.

## [`@model` macro]

`RxInfer` operates with so-called [graphical probabilistic models](https://en.wikipedia.org/wiki/Graphical_model), more specifically [factor graphs](https://en.wikipedia.org/wiki/Factor_graph). Working with graphs directly is, however, tedius and error-prone, especially for large models. To simplify the process, `RxInfer` exports the `@model` macro, which translates a textual description of a probabilistic model into a corresponding factor graph representation.

```@docs
RxInfer.@model
```

Note, that `GraphPPL` also implements `@model` macro, but does **not** export it by default. This was a deliberate choice to allow inference backends (such as `RxInfer`) to implement [custom functionality](@ref lib-model-construction-pipelines) on top of the default `GraphPPL.@model` macro. This is done with a custom  _backend_ for `GraphPPL.@model` macro. Read more about backends in the corresponding section of `GraphPPL` [documentation](https://github.com/ReactiveBayes/GraphPPL.jl).

```@docs
RxInfer.ReactiveMPGraphPPLBackend
```

## [Conditioning on data](@id lib-model-construction-conditioning)

```@docs
RxInfer.condition_on
RxInfer.ConditionedModelGenerator
```

```@docs 
RxInfer.DefferedDataHandler
```

## [Additional `GraphPPL` pipeline stages](@id lib-model-construction-pipelines)

`RxInfer` implements several additional pipeline stages for default parsing stages in `GraphPPL`.
A notable distinction of the `RxInfer` model specification language is the fact that `RxInfer` "folds" 
some mathematical expressions and adds extra brackets to ensure the correct number of arguments for factor nodes.
For example an expression `x ~ x1 + x2 + x3 + x4` becomes `x ~ ((x1 + x2) + x3) + x4` to ensure that the `+` function has exactly two arguments.
 
```@docs 
RxInfer.error_datavar_constvar_randomvar
RxInfer.compose_simple_operators_with_brackets
RxInfer.inject_tilderhs_aliases
RxInfer.ReactiveMPNodeAliases
```
