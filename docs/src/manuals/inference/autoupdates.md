# [Autoupdates specification](@id autoupdates-guide)

```@docs
@autoupdates
```

`RxInfer` supports [streaming inference](@ref manual-online-inference) on infinite datastreams, wherein posterior beliefs over latent states update automatically as soon as new observations are available. However, we also aim to update our priors given updated beliefs. Let's begin with a simple example:
```@example autoupdates-examples
using RxInfer

@model function streaming_beta_bernoulli(a, b, y)
    θ ~ Beta(a, b)
    y ~ Bernoulli(θ)
end
```
For this model, the `RxInfer` engine will update the posterior belief over the variable `θ` every time we receive a new observation `y`. However, we also wish to update our prior belief by adjusting the arguments `a` and `b` as soon as we have a new belief for the variable `θ`. The `@autoupdates` macro automates this process, simplifying the task of writing automatic updates for certain model arguments based on new beliefs within the model.
Here's how it could look:
```@example autoupdates-examples
autoupdates = @autoupdates begin 
    a, b = params(q(θ))
end
```
This specification directs the `RxInfer` inference engine to update `a` and `b` by invoking the `params` function on the posterior `q` of `θ`. The `params` function, defined in the `Distributions.jl` package, extracts the parameters (`a` and `b` in this case) in the form of a tuple of the resulting posterior (`Beta`) distribution.
```@eval
# Change the text above if this test is failing
using RxInfer, Test, Distributions
@test RxInfer.getmappingfn(RxInfer.getmapping(RxInfer.getautoupdate(autoupdates, 1))) === Distributions.params
nothing
```

## General syntax

The `@autoupdates` accepts a block of code or a function definition, where it detects and transforms lines of the following structure
```julia
(model_arguments...) = some_function(model_variables...)
```
to which we refer as the _individual autoupdate specification_. Other expressions are left untouched.

The `model_variables` can be the following
- `q(θ)` - listens to updates from marginal posteriors of the variable `θ` or from entire collection `θ`
- `q(θ[i])` - listens to updates from marginal posteriors of the collection of variables `θ` at index `i`
- Any constant will work just fine, e.g. `a, b = some_function(q(θ), a_constant)`. Note, however, that an individual autoupdate specification should depend on at least on `q(_)`.

!!! warn 
    `q(θ)[i]` syntax is not supported, use `getindex(q(θ), i)` instead.

Individual autoupdate specifications can involve somewhat complex expressions, as demonstrated below:
```@example autoupdates-examples
@autoupdates begin 
    a = mean(q(θ)) / 2
    b = 2 * (mean(q(θ)) + 1)
end
```





The `@autoupdates`  macro accepts a block of code or a function, detects lines of code of the following structure
```julia
(model_arguments...) = f(model_variables...)
```
and converts them to the corresponding auto-update specification. 

Checks if `arguments...` has either `q(_)` in its sub-expressions and adds such expressions to the specification list. 
All other expressions are left untouched. The result of the macro execution is the [`RxInfer.AutoUpdateSpecification`](@ref) structure that holds the collection 
of individual auto-update specifications.

Each individual auto-update specification refers to model's arguments (which need to be updated) on the left hand side of the equality expression and 
the update function on the right hand side of the expression. The update function operates on posterior marginals in the form of the `q(symbol)` expression.

For example:

```@example autoupdates-guide
using RxInfer

@autoupdates begin 
    x = mean(q(z))
end
```

This structure specifies to automatically update argument `x` as soon as the inference engine computes new posterior over `z` variable.
It then applies the `mean` function to the new posterior and updates the value of `x` automatically. 

As another example consider the following model and auto-update specification:

```julia
@model function kalman_filter(y, x_current_mean, x_current_var)
    x_current ~ Normal(mean = x_current_mean, var = x_current_var)
    x_next    ~ Normal(mean = x_current, var = 1.0)
    y         ~ Normal(mean = x_next, var = 1.0)
end
```

This model has two arguments that represent our prior knowledge of the `x_current` state of the system. 
The `x_next` random variable represent the next state of the system that 
is connected to the observed variable `y`. The auto-update specification could look like:

```julia
autoupdates = @autoupdates begin
    x_current_mean, x_current_var = mean_var(q(x_next))
end
```
This structure specifies to update our prior as soon as we have a new posterior `q(x_next)`. It then applies the `mean_var` function on the 
updated posteriors and updates `x_current_mean` and `x_current_var` automatically.

More complex `@autoupdates` are also allowed. For example, the following code is a valid `@autoupdates` specification:
```julia
@autoupdates begin 
    x = clamp(mean(q(z)), 0, 1)
end
```

## The options block

Optionally, the `@autoupdates` macro accepts a set of `[ options... ]` before the main block. The available options are:
- `warn = true/false`: Enables or disables warnings when with incomaptible model. Set to `true` by default.
- `strict = true/false`: Turns warnings into errors. Set to `false` by default.

## Internal data structures

```@docs
RxInfer.AutoUpdateSpecification
RxInfer.parse_autoupdates
RxInfer.autoupdate_check_reserved_expressions
RxInfer.numautoupdates
Base.isempty(specification::RxInfer.AutoUpdateSpecification)
RxInfer.getautoupdate
RxInfer.addspecification
RxInfer.getvarlabels
RxInfer.IndividualAutoUpdateSpecification
RxInfer.getmapping
RxInfer.AutoUpdateVariableLabel
RxInfer.AutoUpdateMapping
RxInfer.AutoUpdateFetchMarginalArgument
RxInfer.AutoUpdateFetchMessageArgument
RxInfer.prepare_autoupdates_for_model
```