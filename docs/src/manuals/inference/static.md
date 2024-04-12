# [Static Inference](@id manual-static-inference)

This guide explains how to use the [`infer`](@ref) function for static datasets. We'll show how `RxInfer` can estimate posterior beliefs given a set of observations. We'll use a simple Beta-Bernoulli model as an example, which has been covered in the [Getting Started](@ref user-guide-getting-started) section, but keep in mind that these techniques can apply to any model.

Also read about [Streaming Inference](@ref manual-online-inference) or checkout more complex [examples](https://reactivebayes.github.io/RxInfer.jl/stable/examples/overview/).

## [Model specification](@id manual-static-inference-model-spec)

Also read the [Model Specification](@ref user-guide-model-specification) section.

In static inference, we want to update our prior beliefs about certain hidden states given some dataset.
To achieve this, we include data as an argument in our model specification:

```@example manual-static-inference
using RxInfer
using Test #hide

@model function beta_bernoulli(y, a, b)
    θ ~ Beta(a, b)  
    for i in 1:length(y)
        y[i] ~ Bernoulli(θ)
    end
end
```

In this model, we assume that `y` is a collection of data points, and `a` and `b` are just numbers.
To run inference in this model, we have to call the [`infer`](@ref) function with the `data` argument provided.

## [Dataset of observations](@id manual-static-inference-dataset)

For demonstration purposes, we will use hand crafted dataset:

```@example manual-static-inference
using Distributions, StableRNGs

hidden_θ       = 1 / 3.1415
distribution   = Bernoulli(hidden_θ)
rng            = StableRNG(43)
n_observations = 1_000
dataset        = rand(rng, distribution, n_observations)
nothing #hide
```

## [Calling the inference procedure](@id manual-static-inference-infer)

Everything is ready to run inference in our simple model. In order to run inference with static dataset using the [`infer`](@ref) function, we need to use the `data` argument. The `data` argument 
expects a `NamedTuple` where keys correspond to the names of the model arguments. In our case the model arguments were `a`, `b` and `y`. We treat `a` and `b` as hyperparameters and pass them directly to the model constructor and we treat `y` as our observations, thus we pass it to the `data` argument as follows:
```@example manual-static-inference
results = infer(
    model = beta_bernoulli(a = 1.0, b = 1.0),
    data  = (y = dataset, )
)
@test results isa InferenceResult #hide
results #hide
```

!!! note
    `y` inside the `@model` specification is not the same data collection as provided in the `data` argument. Inside the `@model`, `y` is a collection of nodes in the corresponding factor graph, but it will have exactly the same shape as the collection provided in the `data` argument, hence we can use some basic Julia function, e.g. `length`. 

Note, that we could also pass `a` and `b` as data:
```@example manual-static-inference
results = infer(
    model = beta_bernoulli(),
    data  = (y = dataset, a = 1.0, b = 1.0)
)
@test results isa InferenceResult #hide
results #hide
```

The [`infer`](@ref) function, however, requires at least one data argument to be present in the supplied `data`. The difference between _hyperparameters_ and _observations_ is purely semantic and should not have real influence on the result of the inference procedure. 

!!! note
    The inference procedure uses __reactive message passing__ protocol and may decide to optimize and precompute certain messages that use fixed hyperparameters, hence changing the order of computed messages. The order of computations may change the convergence properties for some complex models.

In case of inference with static datasets, the [`infer`](@ref) function will return the [`InferenceResult`](@ref) structure. This structure has the `.posteriors` field, which is a `Dict` like structure that maps names of latent states to their corresponding posteriors. For example:
```@example manual-static-inference
@test results.posteriors[:θ] isa Beta #hide
results.posteriors[:θ]
``` 

```@docs
InferenceResult
```

We can also visualize our posterior results with the `Plots.jl` package. We 
used `Beta(a = 1.0, b = 1.0)` as a prior, lets compare our prior and posterior beliefs:
```@example manual-static-inference
using Plots

rθ = range(0, 1, length = 1000)

p = plot()
p = plot!(p, rθ, (x) -> pdf(Beta(1.0, 1.0), x), title="Prior", fillalpha=0.3, fillrange = 0, label="P(θ)", c=1,)
p = plot!(p, rθ, (x) -> pdf(results.posteriors[:θ], x), title="Posterior", fillalpha=0.3, fillrange = 0, label="P(θ|y)", c=3)
```

## [Variational Inference with static datasets](@id manual-static-inference-variational-inference)

The example above is quite simple and performs exact Bayesian inference. However, for more complex model, we may need to specify variational constraints and perform variational inference. To demonstrate this, we will use a slightly more complex model, where we need to estimate mean and the precision of IID samples drawn from the [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution):

```@example manual-static-inference
@model function iid_estimation(y)
    μ  ~ Normal(mean = 0.0, precision = 0.1)
    τ  ~ Gamma(shape = 1.0, rate = 1.0)
    y .~ Normal(mean = μ, precision = τ)
end
```

In this model, we have two latent variables `μ` and `τ` and a set of observations `y`. Note 
that we used the broadcasting syntax, which is roughly equivalent to the manual for loop shown in the previous example. Let's try to run the inference in this model, but first, we need to create our observations:

```@example manual-static-inference
# `ExponentialFamily` package expors different parametrizations 
# for the Normal distribution
using ExponentialFamily

hidden_μ       = 3.1415
hidden_τ       = 2.7182
distribution   = NormalMeanPrecision(hidden_μ, hidden_τ)
rng            = StableRNG(42)
n_observations = 1_000
dataset        = rand(rng, distribution, n_observations)
nothing #hide
```

And finally we run the inference procedure:

```julia
results = infer(
    model = iid_estimation(),
    data  = (y = dataset, )
)
```
```julia
ERROR: Variables [ μ, τ ] have not been updated after an update event. 
Therefore, make sure to initialize all required marginals and messages. See `initialization` keyword argument for the inference function. 
See the official documentation for detailed information regarding the initialization.

Stacktrace:
 [1] error(s::String)
   @ Base ./error.jl:35
```
```@eval
# This is just a test to ensure that the example above does
# indeed fail with the exact same error
using RxInfer, Test 
@model function iid_estimation(y)
    μ  ~ Normal(mean = 0.0, precision = 0.1)
    τ  ~ Gamma(shape = 1.0, rate = 1.0)
    y .~ Normal(mean = μ, precision = τ)
end
@test_throws "Variables [ μ, τ ] have not been updated after an update event." infer(
    model = iid_estimation(),
    data  = (y = rand(10), )
)
nothing
```

Huh? We get an error saying that the inference could not update the latent variables. This is happened because our model contain loops in its structure, therefore it requires the initialization. Read more about the initialization in the [corresponding section](@ref initialization) in the documentation.

We have two options here, either we initialize the messages and perform [Loopy Belief Propagation](https://en.wikipedia.org/wiki/Belief_propagation) in this model or we break the loops with [variational constraints](@ref user-guide-constraints-specification) and perform variational inference. In this tutorial, we will choose the second option. For this we need to specify factorization constraints with the [`@constraints`](@ref) macro.

```@example manual-static-inference
# Specify mean-field constraint over the joint variational posterior
constraints = @constraints begin 
    q(μ, τ) = q(μ)q(τ)
end
# Specify initial posteriors for variational iterations
initialization = @initialization begin 
    q(μ) = vague(NormalMeanPrecision)
    q(τ) = vague(GammaShapeRate)
end
nothing #hide
```

With this, we can use the `constraints` and `initialization` keyword arguments in the [`infer`](@ref) function. We also specify the number of variational iterations with the `iterations` keyword argument:
```@example manual-static-inference
results = infer(
    model          = iid_estimation(),
    data           = (y = dataset, ),
    constraints    = constraints,
    iterations     = 100,
    initialization = initialization
)
```

Nice! Now, we have some result. Let's for example inspect the posterior results for `μ`.

```@example manual-static-inference
results.posteriors[:μ]
```

In constrast to the previous example, now we have an array of posteriors for `μ`, not just a single value. Each posterior in the collection corresponds to the intermediate variational update for each variational iteration. Let's visualize how our posterior over `μ` has been changing during the variational optimization:

```@example manual-static-inference
@gif for (i, intermediate_posterior) in enumerate(results.posteriors[:μ])
    rμ = range(0, 5, length = 1000)
    plot(rμ, (x) -> pdf(intermediate_posterior, x), title="Posterior on iteration $(i)", fillalpha=0.3, fillrange = 0, label="P(μ|y)", c=3)
    vline!([hidden_μ], label = "Real μ")
end
```

It seems that the posterior has converged to a stable distribution pretty fast. 
We are going to verify the converge in the [next section](@ref manual-static-inference-bfe).
If, for example, we are not interested in intermediate updates, but just in the final posterior, we could use the `returnvars` option in the [`infer`](@ref) function and use the [`KeepLast`](@ref) option for `μ`:

```@example manual-static-inference
results_keep_last = infer(
    model          = iid_estimation(),
    data           = (y = dataset, ),
    constraints    = constraints,
    iterations     = 100,
    returnvars     = (μ = KeepLast(), ),
    initialization = initialization
)
```

We can also verify that the got exactly the same result:
```@example manual-static-inference
@test results_keep_last.posteriors[:μ] == last(results.posteriors[:μ]) #hide
results_keep_last.posteriors[:μ] == last(results.posteriors[:μ])
```

Let's also verify that the posteriors are consistent with the real hidden values used in the dataset generation:

```@example manual-static-inference
println("Real μ was ", hidden_μ)
println("Inferred mean for μ is ", mean(last(results.posteriors[:μ])), " with standard deviation ", std(last(results.posteriors[:μ])))

println("Real τ was ", hidden_τ)
println("Inferred mean for τ is ", mean(last(results.posteriors[:τ])), " with standard deviation ", std(last(results.posteriors[:τ])))

@test abs(mean(last(results.posteriors[:μ])) - hidden_μ) < 3std(last(results.posteriors[:μ])) #hide
@test abs(mean(last(results.posteriors[:τ])) - hidden_τ) < 3std(last(results.posteriors[:τ])) #hide
nothing #hide
```

```@example manual-static-inference
rμ = range(2, 4, length = 1000)
pμ = plot(rμ, (x) -> pdf(last(results.posteriors[:μ]), x), title="Posterior for μ", fillalpha=0.3, fillrange = 0, label="P(μ|y)", c=3)
pμ = vline!(pμ, [ hidden_μ ], label = "Real μ")

rτ = range(2, 4, length = 1000)
pτ = plot(rτ, (x) -> pdf(last(results.posteriors[:τ]), x), title="Posterior for τ", fillalpha=0.3, fillrange = 0, label="P(τ|y)", c=3)
pτ = vline!(pτ, [ hidden_τ ], label = "Real τ")

plot(pμ, pτ)
```

Nice result! Our posteriors are pretty close to the actual values of the parameters used for dataset generation.

## [Convergence and Bethe Free Energy](@id manual-static-inference-bfe)

Read also the [Bethe Free Energy](@ref lib-bethe-free-energy) section.

In contrast to Loopy Belief Propagation, the variational inference is set to converge to a stable point during variational inference. In order to verify the convergence for this particular model, we can check the convergence of the [Bethe Free Enegrgy](@ref lib-bethe-free-energy) values. By default, [`infer`](@ref) function does **not** compute the Bethe Free Energy values. In order to compute those, we must set the `free_energy` flag explicitly to `true`:
```@example manual-static-inference
results = infer(
    model          = iid_estimation(),
    data           = (y = dataset, ),
    constraints    = constraints,
    iterations     = 100,
    initialization = initialization,
    free_energy    = true
)
```

Now, we can access the `free_energy` field of the `results` and verify if the inference procedure has converged or not:
```@example manual-static-inference
plot(results.free_energy, label = "Bethe Free Energy")
```

Well, it seems that `100` iterations was too much for this simple problem and we could do much less iterations in order to converge to a stable point. The animation above also suggested that the posterior for `μ` has converged pretty fast to a stable point.

```@example manual-static-inference
# Let's try to use only 5 iterations
results = infer(
    model          = iid_estimation(),
    data           = (y = dataset, ),
    constraints    = constraints,
    iterations     = 5,
    initialization = initialization,
    free_energy    = true
)
```

```@example manual-static-inference
plot(results.free_energy, label = "Bethe Free Energy")
```

## [Callbacks](@id manual-static-inference-callbacks)

The [`infer`](@ref) function has its own lifecycle.
A user is free to inject some extra logic during the inference procedure, e.g. for [debugging purposes](@ref user-guide-debugging-callbacks).
Here are available callbacks that can be used together with the static datasets:
```@eval
using RxInfer, Test, Markdown
# Update the documentation below if this test does not pass
@test RxInfer.available_callbacks(RxInfer.batch_inference) === Val((
    :on_marginal_update,
    :before_model_creation,
    :after_model_creation,
    :before_inference,
    :before_iteration,
    :before_data_update,
    :after_data_update,
    :after_iteration,
    :after_inference
)) 
nothing
```

---

```julia
before_model_creation()
```
Calls before the model is going to be created, does not accept any arguments.

```julia
after_model_creation(model::ProbabilisticModel)
```
Calls right after the model has been created, accepts a single argument, the `model`.

---

Here is an example usage of the outlined callbacks:

```@example manual-static-inference
before_model_creation_called = Ref(false) #hide
after_model_creation_called = Ref(false) #hide
on_marginal_update_called = Ref(false) #hide

function before_model_creation()
    before_model_creation_called[] = true #hide
    println("The model is about to be created")
end

function after_model_creation(model::ProbabilisticModel)
    after_model_creation_called[] = true #hide
    println("The model has been created")
    println("  The number of factor nodes is: ", length(RxInfer.getfactornodes(model)))
    println("  The number of latent states is: ", length(RxInfer.getrandomvars(model)))
    println("  The number of data points is: ", length(RxInfer.getdatavars(model)))
    println("  The number of constants is: ", length(RxInfer.getconstantvars(model)))
end

function on_marginal_update(model::ProbabilisticModel, name, update)
    on_marginal_update_called[] = true #hide
    println("New marginal update for ", name, " ", update)
end
```

```@example manual-static-inference
results = infer(
    model          = iid_estimation(),
    data           = (y = dataset, ),
    constraints    = constraints,
    iterations     = 5,
    initialization = initialization,
    free_energy    = true,
    callbacks      = (
        before_model_creation = before_model_creation,
        after_model_creation = after_model_creation,
        on_marginal_update = on_marginal_update
    )
)

@test before_model_creation_called[] #hide
@test after_model_creation_called[] #hide
@test on_marginal_update_called[] #hide

nothing #hide
```

## [Where to go next?](@id manual-static-inference-where-to-go)

This guide covered some fundamental usages of the [`infer`](@ref) function in the context of inference with static datasets, 
but did not cover all the available keyword arguments of the function. Read more explanation about the other keyword arguments 
in the [Overview](@ref manual-inference-overview) section or check out the [Streaming Inference](@ref manual-online-inference) section.
Also check out more complex [examples](https://reactivebayes.github.io/RxInfer.jl/stable/examples/overview/).