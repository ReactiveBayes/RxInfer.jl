# Migration Guide from version 2.x to 3.x

This guide is intended to help you migrate your project from version 2.x to 3.x of `RxInfer`. The main difference between these two versions is the redefinition of the model specification language. A detailed explanation of the new model definition language can be found in the [GraphPPL documentation](https://reactivebayes.github.io/GraphPPL.jl/dev/migration_3_to_4/). Here, we will give an overview of the most important changes and introduce `RxInfer` specific changes.

## Model Definition

The model definition in the `@model` macro has changed significantly. This change also has implications for the `infer` function. Since all interfaces to a model are now passed as arguments to the `@model` macro, the `infer` function needs additional information on model construction. Therefore we only support keyword arguments on model construction. An example of the new model definition is shown below:

```@example migration-guide
using RxInfer

@model function coin_toss(prior, y)
    θ ~ prior
    y .~ Bernoulli(θ)
end

# Here, we pass a prior as a parameter to the model, and the data `y` is passed as data. Since we have to distinguish between what should be used as which argument, we have to pass the data as a keyword argument.
infer(model = coin_toss(prior=Beta(1, 1)), 
        data=(y=[1, 0, 1],) 
)
```

## Initialization

Initialization of messages and marginals to kickstart the inference procedure was previously done with the `initmessages` and `initmarginals` keyword. With the introduction of a nested model specificiation in the `@model` macro, we now need a more specific way to initialize messages and marginals. This is done with the new `@initialization` macro. The syntax for the `@initialization` macro is similar to the `@constraints` and `@meta` macro. An example is shown below:

```@example migration-guide
@initialization begin
    # Initialize the marginal for the variable x
    q(x) = vague(NormalMeanVariance)

    # Initialize the message for the variable z
    μ(z) = vague(NormalMeanVariance)

    # Specify the initialization for a submodel of type `submodel`
    for init in submodel
        q(some_var) = vague(NormalMeanVariance)
    end

    # Specify the initialization for a submodel of type `submodel` with a specific index
    for init in (submodel, 1)
        q(some_var) = vague(NormalMeanVariance)
    end
end
```

Similar to the `@constraints` macro, the `@initialization` macro also supports function definitions:

```@example migration-guide
@initialization function my_init()
    # Initialize the marginal for the variable x
    q(x) = vague(NormalMeanVariance)

    # Initialize the message for the variable z
    μ(z) = vague(NormalMeanVariance)

    # Specify the initialization for a submodel of type `submodel`
    for init in submodel
        q(some_var) = vague(NormalMeanVariance)
    end

    # Specify the initialization for a submodel of type `submodel` with a specific index
    for init in (submodel, 1)
        q(some_var) = vague(NormalMeanVariance)
    end
end
```

The result of the initialization macro can be passed to the inference function with keyword argument `initialization`.

## Deprecated syntax

The following syntax is deprecated and will be removed in future versions of `RxInfer`:
- `initmessages` and `initmarginals` keyword arguments
- `randomvar` and `datavar` syntax in the `@model` macro