# [Model construction in RxInfer](@id lib-model-construction)

Model creation in `RxInfer` largely depends on [`GraphPPL`](https://github.com/ReactiveBayes/GraphPPL.jl) package.
`RxInfer` re-exports the `@model` macro from `GraphPPL` and defines extra plugins and data structures on top of the default functionality.

!!! note
    The model creation and construction were largely refactored in `GraphPPL` v4. 
    Read [_Migration Guide_](https://reactivebayes.github.io/GraphPPL.jl/stable/migration_3_to_4/) for more details.

Also read the [_Model Specification_](@ref user-guide-model-specification) guide.

## [`@model` macro](@id lib-model-construction-model-macro)

`RxInfer` operates with so-called [graphical probabilistic models](https://en.wikipedia.org/wiki/Graphical_model), more specifically [factor graphs](https://en.wikipedia.org/wiki/Factor_graph). Working with graphs directly is, however, tedious and error-prone, especially for large models. To simplify the process, `RxInfer` exports the `@model` macro, which translates a textual description of a probabilistic model into a corresponding factor graph representation.

```@docs
RxInfer.@model
```

Note, that `GraphPPL` also implements `@model` macro, but does **not** export it by default. This was a deliberate choice to allow inference backends (such as `RxInfer`) to implement [custom functionality](@ref lib-model-construction-pipelines) on top of the default `GraphPPL.@model` macro. This is done with a custom  _backend_ for `GraphPPL.@model` macro. Read more about backends in the corresponding section of `GraphPPL` [documentation](https://github.com/ReactiveBayes/GraphPPL.jl).

```@docs
RxInfer.ReactiveMPGraphPPLBackend
```

## [Conditioning on data](@id lib-model-construction-conditioning)

After model creation `RxInfer` uses [`RxInfer.condition_on`](@ref) function to condition on data. 
As an alias it is also possible to use the `|` operator for the same purpose, but with a nicer syntax.


```@docs
RxInfer.condition_on
Base.:(|)(generator::RxInfer.ModelGenerator, data)
RxInfer.ConditionedModelGenerator
```

Sometimes it might be useful to condition on data, which is not available at model creation time. 
This might be especially useful in [reactive inference](@ref manual-online-inference) setting, where data, e.g. might be available later on from some asynchronous sensor input. For this reason, `RxInfer` implements a special _deferred_ data handler, that does mark model argument as data, but does not specify any particular value for this data nor its shape.

```@docs 
RxInfer.DeferredDataHandler
```

After the model has been conditioned it can be materialized with the [`RxInfer.create_model`](@ref) function.
This function takes the [`RxInfer.ConditionedModelGenerator`](@ref) object and materializes it into a [`RxInfer.ProbabilisticModel`](@ref).

```@docs 
RxInfer.create_model(generator::RxInfer.ConditionedModelGenerator)
RxInfer.ProbabilisticModel
RxInfer.getmodel(model::RxInfer.ProbabilisticModel)
RxInfer.getreturnval(model::RxInfer.ProbabilisticModel)
RxInfer.getvardict(model::RxInfer.ProbabilisticModel)
RxInfer.getrandomvars(model::RxInfer.ProbabilisticModel)
RxInfer.getdatavars(model::RxInfer.ProbabilisticModel)
RxInfer.getconstantvars(model::RxInfer.ProbabilisticModel)
RxInfer.getfactornodes(model::RxInfer.ProbabilisticModel)
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

## [Getting access to an internal variable data structures](@id lib-model-constriction-internal-variable)

To get an access to an internal `ReactiveMP` data structure of a variable in `RxInfer` model, it is possible to return 
a so called _label_ of the variable from the model macro, and access it later on as the following:

```@example internal-access
using RxInfer
using Test #hide

@model function beta_bernoulli(y)
    θ ~ Beta(1, 1)
    y ~ Bernoulli(θ)
    return θ
end

result = infer(
    model = beta_bernoulli(),
    data  = (y = 0.0, )
)
```

```@example internal-access
graph     = RxInfer.getmodel(result.model)
returnval = RxInfer.getreturnval(graph)
θ         = returnval
variable  = RxInfer.getvariable(RxInfer.getvarref(graph, θ))
@test variable isa ReactiveMP.RandomVariable #hide
ReactiveMP.israndom(variable)
```
