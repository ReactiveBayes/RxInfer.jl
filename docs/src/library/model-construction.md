# [Model construction in RxInfer](@id lib-model-construction)

Model creation in `RxInfer` largely depends on [`GraphPPL`](https://github.com/ReactiveBayes/GraphPPL.jl) package.
`RxInfer` re-exports the `@model` macro from `GraphPPL` and defines extra plugins and data structures on top of the default functionality.

!!! note
    The model creation and construction were largely refactored in `GraphPPL` v4. 
    Read [_Migration Guide_](https://reactivebayes.github.io/GraphPPL.jl/stable/migration_3_to_4/) for more details.

Also read the [_Model Specification_](@ref user-guide-model-specification) guide.

```@docs
RxInfer.ReactiveMPGraphPPLBackend
RxInfer.@model
```

## Additional `GraphPPL` pipeline stages 

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
