# Migration Guide from version 2.x to 3.x

This guide is intended to help you migrate your project from version 2.x to 3.x of `RxInfer`. The main difference between these two versions is the redefinition of the model specification language. A detailed explanation of the new model definition language can be found in the [GraphPPL documentation](https://reactivebayes.github.io/GraphPPL.jl/stable/migration_3_to_4/). Here, we will give an overview of the most important changes and introduce `RxInfer` specific changes.

## Model specification

Also read the [Model specification](@ref user-guide-model-specification) guide.

### `randomvar`, `datavar` and `constvar` have been removed

The most notable change in the model specification is the removal of the `randomvar`, `datavar`, and `constvar` functions.
Now, the `@model` macro automatically determines whether to use `randomvar` or `constvar` based on their usage.
Previously declared `datavar` variables must now be listed in the argument list of the model.

The following example is a simple model definition in previous version:
```julia
@model function SSM(n, x0, A, B, Q, P) 
    x = randomvar(n) 
    y = datavar(Vector{Float64}, n) 
    x_prior ~ MvNormal(μ = mean(x0), Σ = cov(x0)) 
    x_prev = x_prior 
    for i in 1:n 
        x[i] ~ MvNormal(μ = A * x_prev, Σ = Q) 
        y[i] ~ MvNormal(μ = B * x[i], Σ = P) 
        x_prev = x[i] 
    end 
end 
```

The equivalent model definition in the new version is as follows:
```julia
@model function SSM(y, prior_x, A, B, Q, P) 
    x_prev ~ prior_x
    for i in eachindex(y)
        x[i] ~ MvNormal(μ = A * x_prev, Σ = Q) 
        y[i] ~ MvNormal(μ = B * x[i], Σ = P) 
        x_prev = x[i]
    end
end
```

Read more about the change in the [GraphPPL documentation](https://reactivebayes.github.io/GraphPPL.jl/stable/migration_3_to_4/) and 
in the updated [Model specification](@ref user-guide-model-specification) guide.

### Positional arguments are converted to keyword arguments

The changes in the model specification also have implications for the [`infer`](@ref) function. Since all interfaces to a model are now passed as arguments to the `@model` macro, the `infer` function needs additional information on model construction. Therefore, the model function definition converts all positional arguments to keyword arguments. Positional arguments are no longer supported in the model function definition. Below is an example of the new model definition:

```@example migration-guide
using Test #hide
using RxInfer

@model function coin_toss(prior, y)
    θ ~ prior
    y .~ Bernoulli(θ)
end

# Here, we pass a prior as a parameter to the model, and the data `y` is passed as data. 
# Since we have to distinguish between what should be used as which argument, we have to pass the data as a keyword argument.
infer(
    model = coin_toss(prior = Beta(1, 1)), 
    data  = (y = [1, 0, 1],) 
)
```

### Multiple dispatch is no longer supported

Due to the previous change, it is not possible to use multiple dispatch for model function definitions. In other words, type constraints for model arguments are ignored because Julia does not support multiple dispatch for keyword arguments.

### Return value from the model function 

Accessing the return value of the model function has changed. Previously, the return value was returned together with the model upon creation. Now, the return value is saved in the model's data structure, which can be accessed with the [`RxInfer.getreturnval`](@ref) function. To demonstrate the difference, previously we could do the following:
```julia
@model function test_model(a, b)
    y = datavar(Float64)
    θ ~ Beta(1.0, 1.0)
    y ~ Bernoulli(θ)
    return "Hello, world!"
end
modelgenerator = test_model(1.0, 1.0)
model, returnval = RxInfer.create_model(modelgenerator)
returnval # "Hello, world!"
```
The new API is changed to:
```@example migration-guide
@model function test_model(y, a, b) #hide
    θ ~ Beta(1.0, 1.0) #hide
    y ~ Bernoulli(θ) #hide
    return "Hello, world!" #hide
end #hide
modelgenerator = test_model(a = 1.0, b = 1.0) | (y = 1, )
model = RxInfer.create_model(modelgenerator)
@test RxInfer.getreturnval(model) == "Hello, world!" #hide
RxInfer.getreturnval(model)
```

The [`InferenceResult`](@ref)  also no longer stores the `returnval` field. Instead, use the `model` field and the [`RxInfer.getreturnval`](@ref) function:
```@example migration-guide
result = infer(
    model = test_model(a = 1.0, b = 1.0),
    data  = (y = 1, )
)
@test RxInfer.getreturnval(result.model) == "Hello, world!" #hide
RxInfer.getreturnval(result.model)
```

### Returning variables from the model

Similar to the previous version, you can still return latent variables from the model definition:
```@example migration-guide
@model function test_model(y, a, b)
    θ ~ Beta(1.0, 1.0)
    y ~ Bernoulli(θ)
    return θ
end
```
However, their type has changed to internal data structures from the `GraphPPL` package. To access the `ReactiveMP` data structures (e.g., to retrieve the messages or marginals streams), use `RxInfer.getvarref` along with `RxInfer.getvariable`:
```@example migration-guide
using ReactiveMP, Rocket
result = infer(
    model = test_model(a = 1.0, b = 1.0),
    data  = (y = 1, )
)

θlabel  = RxInfer.getreturnval(result.model)
θvarref = RxInfer.getvarref(result.model, θlabel)
θvar    = RxInfer.getvariable(θvarref)
@test θvar isa ReactiveMP.RandomVariable #hide
qθ_test = [] #hide
subscribe!(ReactiveMP.getmarginal(θvar) |> take(1), (qθ) -> push!(qθ_test, qθ)) #hide
@test length(qθ_test) === 1 #hide
@test first(ReactiveMP.getdata(qθ_test)) == Beta(2.0, 1.0) #hide

# `|> take(1)` ensures automatic unsubscription 
θmarginals_subscription = subscribe!(ReactiveMP.getmarginal(θvar) |> take(1), (qθ) -> println(qθ))
nothing #hide
```

## Initialization

Initialization of messages and marginals to kickstart the inference procedure was previously done with the `initmessages` and `initmarginals` keyword. With the introduction of a nested model specificiation in the `@model` macro, we now need a more specific way to initialize messages and marginals. This is done with the new [`@initialization`](@ref) macro. 
Read more about the new syntax in the [Initialization](@ref initialization) guide.