# [Autoupdates specification](@id autoupdates-guide)

```@docs
@autoupdates
```

`RxInfer` supports [streaming inference](@ref manual-online-inference) on infinite datastreams, wherein posterior beliefs over latent states update automatically as soon as new observations are available. However, we also aim to update our priors given updated beliefs. Let's begin with a simple example:
```@example autoupdates-examples
using RxInfer
using Test #hide

@model function streaming_beta_bernoulli(a, b, y)
    θ ~ Beta(a, b)
    y ~ Bernoulli(θ)
end
```
For this model, the `RxInfer` engine will update the posterior belief over the variable `θ` every time we receive a new observation `y`. However, we also wish to update our prior belief by adjusting the arguments `a` and `b` as soon as we have a new belief for the variable `θ`. The `@autoupdates` macro automates this process, simplifying the task of writing automatic updates for certain model arguments based on new beliefs within the model. Here's how it could look:
```@example autoupdates-examples
autoupdates = @autoupdates begin 
    a, b = params(q(θ))
end
```
This specification directs the `RxInfer` inference engine to update `a` and `b` by invoking the `params` function on the posterior `q` of `θ`. The `params` function, defined in the `Distributions.jl` package, extracts the parameters (`a` and `b` in this case) in the form of a tuple of the resulting posterior (`Beta`) distribution.
```@eval
# Change the text above if this test is failing
using RxInfer, Test, Distributions
autoupdates = @autoupdates begin 
    a, b = params(q(θ))
end
@test RxInfer.getmappingfn(RxInfer.getmapping(RxInfer.getautoupdate(autoupdates, 1))) === Distributions.params
nothing
```

Now, we can use the `autoupdates` structure in the [`infer`](@ref) function as following:
```@example autoupdates-examples
# The streaming inference supports static datasets as well
data = (y = [ 1, 0, 1 ], )

result = infer(
    model          = streaming_beta_bernoulli(),
    autoupdates    = autoupdates,
    data           = data,
    keephistory    = 3,
    initialization = @initialization(q(θ) = Beta(1, 1))
)
@test result.history[:θ] == [ Beta(2.0, 1.0), Beta(2.0, 2.0), Beta(3.0, 2.0) ] #hide

result.history[:θ]
```
In this example, we also used the [initialization](@ref initialization) keyword argument. 
This is required for latent states, which are used in the `@autoupdates` specification together with streaming inference.

Consider another example with the following model and auto-update specification:

```@example autoupdates-examples
@model function kalman_filter(y, x_current_mean, x_current_var)
    x_current ~ Normal(mean = x_current_mean, var = x_current_var)
    x_next    ~ Normal(mean = x_current, var = 1.0)
    y         ~ Normal(mean = x_next, var = 1.0)
end
```

This model comprises two arguments representing our prior knowledge of the `x_current` state of the system. 
The latent state `x_next` represents the subsequent state of the system, linked to the observed variable `y`. 
An auto-update specification could resemble the following:
```@example autoupdates-examples
autoupdates = @autoupdates begin
    x_current_mean = mean(q(x_next))
    x_current_var  = var(q(x_next))
end
```
This structure dictates updating our prior immediately upon obtaining a new posterior `q(x_next)`. 
It then applies the `mean` and `var` functions to the updated posteriors, thereby automatically updating `x_current_mean` and `x_current_var`.

```@example autoupdates-examples
result = infer(
    model = kalman_filter(),
    data  = (y = rand(3), ),
    autoupdates = autoupdates,
    initialization = @initialization(q(x_next) = NormalMeanVariance(0, 1)),
    keephistory = 3,
)
result.history[:x_next]
```

Read more about streaming inference in the [Streaming (online) inference](@ref manual-online-inference) section.

## General syntax

The `@autoupdates` macro accepts either a block of code or a full function definition. It detects and transforms lines structured as follows:
```julia
(model_arguments...) = some_function(model_variables...)
```
These lines are referred to as _individual autoupdate specifications_. Other expressions remain unchanged. 
The result of the macro execution is the [`RxInfer.AutoUpdateSpecification`](@ref) structure that holds the collection 
of [`RxInfer.IndividualAutoUpdateSpecification`](@ref).

The `@autoupdates` macro identifies an individual autoupdate specification if the `model_variables...` contains:
- `q(s)`, which monitors updates from marginal posteriors of an individual variable `s` or a collection of variables `s`.
- `q(s[i])`, which monitors updates from marginal posteriors of the collection of variables `s` at index `i`.

Expressions not meeting the above criteria remain unmodified. For instance, an expression like `a = f(1)` is not considered an individual autoupdate. Therefore, the `@autoupdates` macro can contain arbitrary expressions and allows for the definition of temporary variables or even functions. Additionally, within an individual autoupdate specification, it is possible to use any intermediate constants, such as `a, b = some_function(q(s), a_constant)`.

The `model_arguments...` can either be a single model argument or a tuple of model arguments, as defined within the `@model` macro. However, it's important to note that if `model_arguments...` is a tuple, for example in `a, b = some_function(q(s))`, then `some_function` must also return a tuple of the same length (of length 2 in this example).

Individual autoupdate specifications can involve somewhat complex expressions, as demonstrated below:
```@example autoupdates-examples
@autoupdates begin 
    a = mean(q(θ)) / 2
    b = 2 * (mean(q(θ)) + 1)
end
```
or
```@example autoupdates-examples
@autoupdates begin 
    x = clamp(mean(q(z)), 0, 1)
end
```

!!! warning
    `q(θ)[i]` or `f(q(θ))[i]` syntax is not supported, use `getindex(q(θ), i)` or `getindex(f(q(θ)), i)` instead.

An individual autoupdate can also simultaneously depend on multiple latent states, e.g:
```@example autoupdates-examples
f(args...) = nothing #hide
g(args...) = nothing #hide
@autoupdates begin 
    a = f(q(μ), q(s), q(τ))
    b = g(q(θ))
end
```

As mentioned before, the `@autoupdates` accepts a full function definition, which can also accepts arbitrary arguments:
```@example autoupdates-examples
@autoupdates function generate_autoupdates(f, condition)
    if condition 
        a = f(q(θ))
    else
        a = f(q(s))
    end
end

autoupdates = generate_autoupdates(mean, true)
```

## The options block

Optionally, the `@autoupdates` macro accepts a set of `[ options... ]` before the main block or the full function definition. The available options are:
- `warn = true/false`: Enables or disables warnings when with incomaptible model. Set to `true` by default.
- `strict = true/false`: Turns warnings into errors. Set to `false` by default.

```@example autoupdates-examples
autoupdates = @autoupdates [ strict = true ] begin 
    a, b = params(q(θ))
end
```
or 
```@example autoupdates-examples
@autoupdates [ strict = true ] function generate_autoupdates()
    a, b = params(q(θ))
end
autoupdates = generate_autoupdates()
```


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