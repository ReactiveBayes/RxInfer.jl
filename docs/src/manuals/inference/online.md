# [Online inference](@id manual-online-inference)

This guide explains how to use the [`infer`](@ref) function for dynamic datasets. We'll show how `RxInfer` can continuously update beliefs asynchronously whenever a new observation arrives. We'll use a simple Beta-Bernoulli model as an example, which has been covered in the [Getting Started](@ref user-guide-getting-started) section, 
but keep in mind that these techniques can apply to any model.

## [Model specification](@id manual-online-inference-model-spec)

In online inference, we want to continuously update our prior beliefs about certain hidden states. 
To achieve this, we include extra arguments in our model specification to allow for dynamic prior changes:

```@example manual-online-inference
using RxInfer
using Test #hide

@model function beta_bernoulli_online(y, a, b)
    θ ~ Beta(a, b)  
    y ~ Bernoulli(θ)
end
```

In this model, we assume we only have one observation y at a time, and the a and b parameters are not fixed to specific values but rather are arguments of the model itself.

## [Automatic prior update](@id manual-online-inference-autoupdates)

Next, we want to enable `RxInfer` to automatically update the `a` and `b` parameters as soon as a new posterior for `θ` is available. To accomplish this, we utilize the [`@autoupdates`](@ref) macro.

```@example manual-online-inference
beta_bernoulli_autoupdates = @autoupdates begin 
    # We want to update `a` and `b` to be equal to the parameters 
    # of the current posterior for `θ`
    a, b = params(q(θ))
end
```

This specification instructs `RxInfer` to update `a` and `b` parameters automatically as as soon as new posterior for `θ` is available.

```@docs
@autoupdates
```

## [Asynchronous data stream of observations](@id manual-online-inference-async-datastream)

For demonstration purposes, we will use hand crafted stream of observations using the `Rocket.jl` library:

```@example manual-online-inference
using Rocket, Distributions, StableRNGs

hidden_θ     = 1 / 3.1415
distribution = Bernoulli(hidden_θ)
rng          = StableRNG(43)
datastream   = RecentSubject(Bool)
```

The [`infer`](@ref) function expects the `datastream` to emit values in the form of the `NamedTuple`s. To simply this process, `Rocket` exports `labeled` function:

```@example manual-online-inference
# We convert the stream of `Bool` to the stream of `Float64`, because the `Bernoulli` node
# from `ReactiveMP` expects `Float64` inputs
observations = labeled(Val((:y, )), combineLatest(datastream |> map(Float64, float)))
```

Let's verify that our datastream, does indeed produce `NamedTuple`

```@example manual-online-inference
test_values = [] #hide
test_subscription = subscribe!(observations, (new_observation) -> push!(test_values, new_observation)) #hide
subscription = subscribe!(observations, 
    (new_observation) -> println("Got new observation `", new_observation, "` 🎉")
)
```

```@example manual-online-inference
for i in 1:5
    next!(datastream, rand(rng, distribution))
end
@test length(test_values) === 5 #hide
@test all(value -> haskey(value, :y) && (isone(value[:y]) || iszero(value[:y])), test_values) #hide 
nothing #hide
```

Nice! Our data stream produces events in a form of the `NamedTuple`s, which is compatible with the [`infer`](@ref) function.

```@example manual-online-inference
unsubscribe!(test_subscription) #hide
# It is important to keep track of the existing susbcriptions
# and unsubscribe to reduce the usage of computational resources
unsubscribe!(subscription)
```

## [Instantiating the reactive inference engine](@id manual-online-inference-inst-reactive-engine)

Now, we have everything ready to start running the inference with `RxInfer` on dynamic datasets with the [`infer`](@ref) function:

```@example manual-online-inference
engine = infer(
    model         = beta_bernoulli_online(),
    datastream    = observations,
    autoupdates   = beta_bernoulli_autoupdates,
    initmarginals = (θ = Beta(1, 1), ),
    autostart     = false
)
```

In the code above, there are several notable differences compared to running inference for static datasets. Firstly, we utilized the `autoupdates` argument as discussed [previously](@ref manual-online-inference-autoupdates). Secondly, we employed the `initmarginals` function to initialize the posterior over `θ`. This is necessary for the `@autoupdates` macro, as it needs to initialize the `a` and `b` parameters before the data becomes available. Thirdly, we set `autostart = false` to indicate that we do not want to immediately subscribe to the datastream, but rather do so manually later using the [`RxInfer.start`](@ref) function.

```@docs
RxInferenceEngine
RxInfer.start
RxInfer.stop
```

Given the `engine`, we now can subscribe on the posterior updates:

```@example manual-online-inference
θ_updates_for_testing_the_example  = [] #hide
θ_updates_for_testing_subscription = subscribe!(engine.posteriors[:θ], (new_posterior_for_θ) -> push!(θ_updates_for_testing_the_example, new_posterior_for_θ)) #hide
θ_updates_subscription = subscribe!(engine.posteriors[:θ], 
    (new_posterior_for_θ) -> println("A new posterior for θ is `", new_posterior_for_θ, "` 🤩")
)
nothing #hide
```

In this setting, we should get a message every time a new posterior is available for `θ`. Let's try to generate a new observation!

```@example manual-online-inference
next!(datastream, rand(rng, distribution))
@test isempty(θ_updates_for_testing_the_example) #hide
nothing #hide
```

Hmm, nothing happened...? Oh, we forgot to _start_ the engine with the [`RxInfer.start`](@ref) function. Let's do that now:

```@example manual-online-inference
RxInfer.start(engine)
@test length(θ_updates_for_testing_the_example) === 1 #hide
nothing #hide
```

Ah, as soon as we start our engine, we receive the posterior for `θ`. This occurred because we initialized our stream as `RecentSubject`, which retains the most recent value and emits it upon subscription. Our engine automatically subscribed to the observations and obtained the most recent value, initiating inference. Let's see if we can add more observations:

```@example manual-online-inference
next!(datastream, rand(rng, distribution))
@test length(θ_updates_for_testing_the_example) === 2 #hide
nothing #hide
```

Great! We got another posterior! Let's try a few more observations:

```@example manual-online-inference
for i in 1:5
    next!(datastream, rand(rng, distribution))
end
@test length(θ_updates_for_testing_the_example) === 7 #hide
nothing #hide
```

As demonstrated, the reactive engine reacts to new observations and performs inference as soon as a new observation is available. But what if we want to maintain a history of posteriors? The [`infer`](@ref) function supports the `historyvars` and `keephistory` arguments precisely for that purpose.
In the next section we reinstantiate our engine, with the `keephistory` argument enabled, but first, we must shutdown the previous engine and unsubscribe from its posteriors:

```@example manual-online-inference
RxInfer.stop(engine)
unsubscribe!(θ_updates_subscription)
unsubscribe!(θ_updates_for_testing_subscription) #hide
nothing #hide
```

## [Keeping the history of posteriors](@id manual-online-inference-history)

To retain the history of posteriors within the engine, we can utilize the `keephistory` and `historyvars` arguments.
The `keephistory` parameter specifies the length of the circular buffer for storing the history of posterior updates, while `historyvars` determines what variables to save in the history and how often to save them (e.g., every iteration or only at the end of iterations).

```@example manual-online-inference
engine = infer(
    model         = beta_bernoulli_online(),
    datastream    = observations,
    autoupdates   = beta_bernoulli_autoupdates,
    initmarginals = (θ = Beta(1, 1), ),
    keephistory   = 100,
    historyvars   = (θ = KeepLast(), ),
    autostart     = true
)
```

In the example above, we specified that we want to store at most `100` posteriors for `θ`, and `KeepLast()` indicates that we are only interested in the final value of `θ` and not in intermediate values during variational iterations. We also specified the `autostart = true` to start the engine automatically without need for [`RxInfer.start`](@ref) and [`RxInfer.stop`](@ref).

!!! note
    In this model, we do not utilize the `iterations` argument, indicating that we perform a single VMP iteration. If multiple iterations were employed, `engine.posteriors[:θ]` would emit every intermediate value.

Now, we can feed some more observations to the datastream:

```@example manual-online-inference
for i in 1:5
    next!(datastream, rand(rng, distribution))
end
```

And inspect the `engine.history[:θ]` buffer:

```@example manual-online-inference
@test length(engine.history[:θ]) === 6 #hide
engine.history[:θ]
```

As we can see the buffer correctly saved the posteriors in the `.history` buffer.

!!! note 
    We have `6` entries, despite having only `5` new observations. As mentioned earlier, this occurs because we initialized our datastream as a `RecentSubject`, which retains the most recent observation and emits it each time a new subscription occurs.

### [Visualizing the history of posterior estimation](@id manual-online-inference-history-visualization)

Let's feed more observation and visualize how the posterior changes over time:

```@example manual-online-inference
for i in 1:94
    next!(datastream, rand(rng, distribution))
end
@test length(engine.history[:θ]) === 100 #hide
nothing #hide
```

```@example manual-online-inference
using Plots

@gif for posterior in engine.history[:θ]
    rθ = range(0, 1, length = 1000)
    pθ = plot(rθ, (x) -> pdf(posterior, x), title="Posterior", fillalpha=0.3, fillrange = 0, label="P(θ|y)", c=3)
    pθ = vline!(pθ, [ hidden_θ ], label = "Real value of θ")

    plot(pθ)
end
```

We can keep feeding data to our datastream, but only last `100` posteriors will be saved in the `history` buffer:

```@example manual-online-inference
for i in 1:200
    next!(datastream, rand(rng, distribution))
end
@test length(engine.history[:θ]) === 100 #hide
nothing #hide
```

```@example manual-online-inference
@gif for posterior in engine.history[:θ]
    rθ = range(0, 1, length = 1000)
    pθ = plot(rθ, (x) -> pdf(posterior, x), title="Posterior", fillalpha=0.3, fillrange = 0, label="P(θ|y)", c=3)
    pθ = vline!(pθ, [ hidden_θ ], label = "Real value of θ")

    plot(pθ)
end
```

!!! note
    It is also possible to visualize the inference estimation continously with manual subscription to `engine.posteriors[:θ]`.

