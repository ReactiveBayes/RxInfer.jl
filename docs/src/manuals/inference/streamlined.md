# [Streaming (online) inference](@id manual-online-inference)

This guide explains how to use the [`infer`](@ref) function for dynamic datasets. We show how `RxInfer` can continuously update beliefs asynchronously whenever a new observation arrives. We use a simple Beta-Bernoulli model as an example, which has been covered in the [Getting Started](@ref user-guide-getting-started) section, 
however, these techniques can be applied to any model

Also read about [Static Inference](@ref manual-static-inference) or checkout more complex [examples](https://examples.rxinfer.com/).

## [Model specification](@id manual-online-inference-model-spec)

Also read the [Model Specification](@ref user-guide-model-specification) section.

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

In this model, we assume we only have one observation y at a time, and the `a` and `b` parameters are not fixed to specific values but rather are arguments of the model itself.

## [Automatic prior update](@id manual-online-inference-autoupdates)

Next, we want to enable `RxInfer` to automatically update the `a` and `b` parameters as soon as a new posterior for `θ` is available. To accomplish this, we utilize the [`@autoupdates`](@ref) macro.

```@example manual-online-inference
beta_bernoulli_autoupdates = @autoupdates begin 
    # We want to update `a` and `b` to be equal to the parameters 
    # of the current posterior for `θ`
    a, b = params(q(θ))
end
```

This specification instructs `RxInfer` to update `a` and `b` parameters automatically as as soon as a new posterior for `θ` is available.
Read more about `@autoupdates` in the [Autoupdates guide](@ref autoupdates-guide)

## [Asynchronous data stream of observations](@id manual-online-inference-async-datastream)

For demonstration purposes, we use a handcrafted stream of observations with the `Rocket.jl` library
```@example manual-online-inference
using Rocket, Distributions, StableRNGs

hidden_θ     = 1 / 3.1415
distribution = Bernoulli(hidden_θ)
rng          = StableRNG(43)
datastream   = RecentSubject(Bool)
```

The [`infer`](@ref) function expects the `datastream` to emit values in the form of the `NamedTuple`s. To simplify this process, `Rocket.jl` exports `labeled` function. We also use the `combineLatest` function to convert a stream of `Bool`s to a stream of `Tuple{Bool}`s. Read more about these function in the [documentation to `Rocket.jl`](https://reactivebayes.github.io/Rocket.jl/stable/).

```@example manual-online-inference
observations = labeled(Val((:y, )), combineLatest(datastream))
```

Let's verify that our datastream does indeed produce `NamedTuple`s

```@example manual-online-inference
test_values = [] #hide
test_subscription = subscribe!(observations, (new_observation) -> push!(test_values, new_observation)) #hide
subscription = subscribe!(observations, 
    (new_observation) -> println("Got new observation ", new_observation, " 🎉")
)
nothing #hide
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
    model          = beta_bernoulli_online(),
    datastream     = observations,
    autoupdates    = beta_bernoulli_autoupdates,
    returnvars     = (:θ, ),
    initialization = @initialization(q(θ) = Beta(1, 1)),
    autostart      = false
)
```

In the code above, there are several notable differences compared to running inference for static datasets. Firstly, we utilized the `autoupdates` argument as discussed [previously](@ref manual-online-inference-autoupdates). Secondly, we employed the [`@initialization`](@ref) macro to initialize the posterior over `θ`. This is necessary for the `@autoupdates` macro, as it needs to initialize the `a` and `b` parameters before the data becomes available. Thirdly, we set `autostart = false` to indicate that we do not want to immediately subscribe to the datastream, but rather do so manually later using the [`RxInfer.start`](@ref) function. The `returnvars` specification differs a little from [Static Inference](@ref manual-static-inference). In reactive inference, the `returnvars = (:θ, )` must be a tuple of `Symbol`s and specifies that we would be interested to get a stream of posteriors update for `θ`. The `returnvars` specification is optional and the inference engine will create reactive streams for all latent states if ommited.

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
    (new_posterior_for_θ) -> println("A new posterior for θ is ", new_posterior_for_θ, " 🤩")
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
    model          = beta_bernoulli_online(),
    datastream     = observations,
    autoupdates    = beta_bernoulli_autoupdates,
    initialization = @initialization(q(θ) = Beta(1, 1)),
    keephistory    = 100,
    historyvars    = (θ = KeepLast(), ),
    autostart      = true
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

To visualize the history of posteriors we use the `@gif` macro from the `Plots` package:
```@example manual-online-inference
using Plots

@gif for posterior in engine.history[:θ]
    rθ = range(0, 1, length = 1000)
    pθ = plot(rθ, (x) -> pdf(posterior, x), fillalpha=0.3, fillrange = 0, label="P(θ|y)", c=3)
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
@gif for posterior in engine.history[:θ]
    rθ = range(0, 1, length = 1000)
    pθ = plot(rθ, (x) -> pdf(posterior, x), fillalpha=0.3, fillrange = 0, label="P(θ|y)", c=3)
    pθ = vline!(pθ, [ hidden_θ ], label = "Real value of θ")

    plot(pθ)
end
```

!!! note
    It is also possible to visualize the inference estimation continously with manual subscription to `engine.posteriors[:θ]`.


As previously it is important to shutdown the inference engine when it becomes unnecessary:
```@example manual-online-inference
RxInfer.stop(engine)
```

## [Subscribing on the stream of free energy](@id manual-online-inference-free-energy)


To obtain a continuous stream of updates for the [Bethe Free Energy](@ref lib-bethe-free-energy), we need to initialize the engine with the `free_energy` argument set to `true`:

```@example manual-online-inference
engine = infer(
    model          = beta_bernoulli_online(),
    datastream     = observations,
    autoupdates    = beta_bernoulli_autoupdates,
    initialization = @initialization(q(θ) = Beta(1, 1)),
    keephistory    = 5,
    autostart      = true,
    free_energy    = true
)
```

!!! note 
    It's important to use the `keephistory` argument alongside the `free_energy` argument because setting `free_energy = true` also maintains an internal circular buffer to track its previous updates.

```@example manual-online-inference
free_energy_for_testing = [] #hide
free_energy_for_testing_subscription = subscribe!(engine.free_energy, (v) -> push!(free_energy_for_testing, v)) #hide
free_energy_subscription = subscribe!(engine.free_energy, 
    (bfe_value) -> println("New value of Bethe Free Energy has been computed ", bfe_value, " 👩‍🔬")
)
@test length(free_energy_for_testing) === 1 #hide
nothing #hide
```

Let's emit more observations:

```@example manual-online-inference
for i in 1:5
    next!(datastream, rand(rng, distribution))
end
@test length(free_energy_for_testing) === 6 #hide
nothing #hide
```

In this particular example, we do not perform any variational iterations and do not use any variational constraints, hence, the inference is exact.
In this case the BFE values are equal to the minus log-evidence of the model given new observation. 
We can also track history of Bethe Free Energy values with the following fields of the `engine`:
- `free_energy_history`: free energy history, averaged across variational iterations value for all observations  
- `free_energy_raw_history`: free energy history, returns returns computed values of all variational iterations for each data event (if available)
- `free_energy_final_only_history`: free energy history, returns computed values of final variational iteration for each data event (if available)

```@example manual-online-inference
@test length(engine.free_energy_history) === 1 #hide
engine.free_energy_history
```

```@example manual-online-inference
@test length(engine.free_energy_raw_history) === 5 #hide
engine.free_energy_raw_history
```

```@example manual-online-inference
@test length(engine.free_energy_final_only_history) === 5 #hide
engine.free_energy_final_only_history
```

```@example manual-online-inference
unsubscribe!(free_energy_for_testing_subscription) #hide
# Stop the engine when not needed as usual
RxInfer.stop(engine)
unsubscribe!(free_energy_subscription)
```

As has been mentioned, in this particular example we do not perform variational iterations, hence, there is little different between different representations of the BFE history buffers. However, when performing variational inference with the `iterations` argument, those buffers will be different. To demonstrate this difference let's build a slightly more complex model with variational constraints:

```@example manual-online-inference
@model function iid_normal(y, mean_μ, var_μ, shape_τ, rate_τ)
    μ ~ Normal(mean = mean_μ, var = var_μ)
    τ ~ Gamma(shape = shape_τ, rate = rate_τ)
    y ~ Normal(mean = μ, precision = τ)
end

iid_normal_constraints = @constraints begin
    q(μ, τ) = q(μ)q(τ)
end

iid_normal_autoupdates = @autoupdates begin 
    mean_μ  = mean(q(μ))
    var_μ   = var(q(μ))
    shape_τ = shape(q(τ))
    rate_τ  = rate(q(τ))
end

iid_normal_hidden_μ       = 3.1415
iid_normal_hidden_τ       = 0.0271
iid_normal_distribution   = NormalMeanPrecision(iid_normal_hidden_μ, iid_normal_hidden_τ)
iid_normal_rng            = StableRNG(123)
iid_normal_datastream     = RecentSubject(Float64)
iid_normal_observations   = labeled(Val((:y, )), combineLatest(iid_normal_datastream))
iid_normal_initialization = @initialization begin 
    q(μ) = NormalMeanPrecision(0.0, 0.001)
    q(τ) = GammaShapeRate(10.0, 10.0)
end

iid_normal_engine  = infer(
    model          = iid_normal(),
    datastream     = iid_normal_observations,
    autoupdates    = iid_normal_autoupdates,
    constraints    = iid_normal_constraints,
    initialization = iid_normal_initialization,
    historyvars    = (
        μ = KeepLast(),
        τ = KeepLast(),
    ),
    keephistory    = 100,
    iterations     = 10,
    free_energy    = true,
    autostart      = true
)
```

The notable differences with the previous example is the use of the `constraints` and `iterations` arguments. Read more about constraints in the [Constraints Specification](@ref user-guide-constraints-specification) section of the documentation. We have also indicated in the `historyvars` that we want to keep track of posteriors only from the last variational iteration in the history buffer.

Now we can feed some observations to the datastream:
```@example manual-online-inference
for i in 1:100
    next!(iid_normal_datastream, rand(iid_normal_rng, iid_normal_distribution))
end
```

Let's inspect the differences in the `free_energy` buffers:

```@example manual-online-inference
@test all(v -> v <= 0.0, diff(iid_normal_engine.free_energy_history)) #hide
@test length(iid_normal_engine.free_energy_history) === 10 #hide
iid_normal_engine.free_energy_history
```

```@example manual-online-inference
@test length(iid_normal_engine.free_energy_raw_history) === 1000 #hide
iid_normal_engine.free_energy_raw_history
```

```@example manual-online-inference
@test length(iid_normal_engine.free_energy_final_only_history) === 100 #hide
iid_normal_engine.free_energy_final_only_history
```

We can also visualize different representations:
```@example manual-online-inference
plot(iid_normal_engine.free_energy_history, label = "Bethe Free Energy (averaged)")
```

!!! note
    In general, the _averaged_ Bethe Free Energy values must decrease and converge to a stable point.

```@example manual-online-inference
plot(iid_normal_engine.free_energy_raw_history, label = "Bethe Free Energy (raw)")
```

```@example manual-online-inference
plot(iid_normal_engine.free_energy_final_only_history, label = "Bethe Free Energy (last per observation)")
```

As we can see, in the case of the variational iterations those buffers are quite different and represent different representations
of the same Bethe Free Energy stream (which corresponds to the `.free_energy_raw_history`). As a sanity check, we could also visualize the history of our posterior estimations in the same way 
as we did for a simpler previous example:

```@example manual-online-inference
@test length(iid_normal_engine.history[:μ]) === 100 #hide
@test length(iid_normal_engine.history[:τ]) === 100 #hide
@gif for (μ_posterior, τ_posterior) in zip(iid_normal_engine.history[:μ], iid_normal_engine.history[:τ])
    rμ = range(0, 10, length = 1000)
    rτ = range(0, 1, length = 1000)

    pμ = plot(rμ, (x) -> pdf(μ_posterior, x), fillalpha=0.3, fillrange = 0, label="P(μ|y)", c=3)
    pμ = vline!(pμ, [ iid_normal_hidden_μ ], label = "Real value of μ")

    pτ = plot(rτ, (x) -> pdf(τ_posterior, x), fillalpha=0.3, fillrange = 0, label="P(τ|y)", c=3)
    pτ = vline!(pτ, [ iid_normal_hidden_τ ], label = "Real value of τ")

    plot(pμ, pτ, layout = @layout([ a; b ]))
end
```

Nice, the history of the estimated posteriors aligns well with the real (hidden) values of the underlying parameters.

## [Callbacks](@id manual-online-inference-callbacks)

The [`RxInferenceEngine`](@ref) has its own lifecycle. The callbacks differ a little bit from [Using callbacks with Static Inference](@ref manual-static-inference-callbacks). Callbacks can be a `NamedTuple`, `Dict`, or any custom structure that implements `ReactiveMP.handle_event`. For a comprehensive overview of the callbacks system, including custom callback structures and model metadata, see the [Callbacks](@ref manual-inference-callbacks) section.
Here are available callbacks that can be used together with the streaming inference:
```@eval
using RxInfer, Test, Markdown
# Update the documentation below if this test does not pass
@test RxInfer.available_callbacks(RxInfer.streaming_inference) === Val((
    :before_model_creation,
    :after_model_creation,
    :before_autostart,
    :after_autostart,
    :before_message_rule_call,
    :after_message_rule_call,
    :before_product_of_two_messages,
    :after_product_of_two_messages,
    :before_product_of_messages,
    :after_product_of_messages,
    :before_form_constraint_applied,
    :after_form_constraint_applied,
    :before_marginal_computation,
    :after_marginal_computation
))
nothing
```

---

Below we list `RxInfer` specific callbacks. In addition to these, `ReactiveMP` provides lower-level callbacks for the message passing procedure itself, such as `before_message_rule_call`, `after_message_rule_call`, `before_product_of_messages`, `after_product_of_messages`, `before_marginal_computation`, `after_marginal_computation`, and others. For a full list and detailed descriptions of these callbacks, refer to the official documentation of `ReactiveMP`.

```julia
before_model_creation(event::BeforeModelCreationEvent)
```
Calls before the model is going to be created. The event has no fields.

```julia
after_model_creation(event::AfterModelCreationEvent)
```
Calls right after the model has been created. Access the model via `event.model`.

```julia
before_autostart(event::BeforeAutostartEvent)
```
Calls before the `RxInfer.start()` function, if `autostart` is set to `true`. Access the engine via `event.engine`.

```julia
after_autostart(event::AfterAutostartEvent)
```
Calls after the `RxInfer.start()` function, if `autostart` is set to `true`. Access the engine via `event.engine`.

---

Here is an example usage of the outlined callbacks:

```@example manual-online-inference
before_model_creation_called = Ref(false) #hide
after_model_creation_called = Ref(false) #hide
before_autostart_called = Ref(false) #hide
after_autostart_called = Ref(false) #hide

function before_model_creation(event::BeforeModelCreationEvent)
    before_model_creation_called[] = true #hide
    println("The model is about to be created")
end

function after_model_creation(event::AfterModelCreationEvent)
    after_model_creation_called[] = true #hide
    println("The model has been created")
    println("  The number of factor nodes is: ", length(RxInfer.getfactornodes(event.model)))
    println("  The number of latent states is: ", length(RxInfer.getrandomvars(event.model)))
    println("  The number of data points is: ", length(RxInfer.getdatavars(event.model)))
    println("  The number of constants is: ", length(RxInfer.getconstantvars(event.model)))
end

function before_autostart(event::BeforeAutostartEvent)
    before_autostart_called[] = true #hide
    println("The reactive inference engine is about to start")
end

function after_autostart(event::AfterAutostartEvent)
    after_autostart_called[] = true #hide
    println("The reactive inference engine has been started")
end

engine = infer(
    model          = beta_bernoulli_online(),
    datastream     = observations,
    autoupdates    = beta_bernoulli_autoupdates,
    initialization = @initialization(q(θ) = Beta(1, 1)),
    keephistory    = 5,
    autostart      = true,
    free_energy    = true,
    callbacks      = (
        before_model_creation = before_model_creation,
        after_model_creation  = after_model_creation,
        before_autostart      = before_autostart,
        after_autostart       = after_autostart
    )
)

@test before_model_creation_called[] #hide
@test after_model_creation_called[] #hide
@test before_autostart_called[] #hide
@test after_autostart_called[] #hide

RxInfer.stop(engine) #hide
nothing #hide
```


## [Event loop](@id manual-online-inference-event-loop)

In constrast to [Static Inference](@ref manual-static-inference), the streaming version of the [`infer`](@ref) function 
does not provide callbacks such as `on_marginal_update`, since it is possible to subscribe directly on those updates with the 
`engine.posteriors` field. However, the reactive inference engine provides an ability to listen to its internal event loop, that also includes "pre" and "post" events for posterior updates.

```@docs
RxInferenceEvent
```

Let's build a simple example by implementing our own event listener that does not do anything complex but simply prints some debugging information.

```@eval
using RxInfer, Test, Markdown
# Update the documentation below if this test does not pass
@test RxInfer.available_events(RxInfer.streaming_inference) === Val((
    :before_start,
    :after_start,
    :before_stop,
    :after_stop,
    :on_new_data,
    :before_iteration,
    :before_auto_update,
    :after_auto_update,
    :before_data_update,
    :after_data_update,
    :after_iteration,
    :before_history_save,
    :after_history_save,
    :on_tick,
    :on_error,
    :on_complete
))
nothing
```

```@example manual-online-inference
struct MyEventListener <: Rocket.Actor{RxInferenceEvent}
    # ... extra fields
end
```

The available events are

```julia
:before_start
```
Emits right before starting the engine with the [`RxInfer.start`](@ref) function.
The data is `(engine::RxInferenceEngine, )`

```@example manual-online-inference
function Rocket.on_next!(listener::MyEventListener, event::RxInferenceEvent{ :before_start })
    (engine, ) = event
    @test engine isa RxInferenceEngine #hide
    println("The engine is about to start.")
end
```

```julia
:after_start
```
Emits right after starting the engine with the [`RxInfer.start`](@ref) function.
The data is `(engine::RxInferenceEngine, )`

```@example manual-online-inference
function Rocket.on_next!(listener::MyEventListener, event::RxInferenceEvent{ :after_start })
    (engine, ) = event
    @test engine isa RxInferenceEngine #hide
    println("The engine has been started.")
end
```

```julia
:before_stop
```
Emits right before stopping the engine with the [`RxInfer.stop`](@ref) function.
The data is `(engine::RxInferenceEngine, )`

```@example manual-online-inference
function Rocket.on_next!(listener::MyEventListener, event::RxInferenceEvent{ :before_stop })
    (engine, ) = event
    @test engine isa RxInferenceEngine #hide
    println("The engine is about to be stopped.")
end
```

```julia
:after_stop
```
Emits right after stopping the engine with the [`RxInfer.stop`](@ref) function.
The data is `(engine::RxInferenceEngine, )`

```@example manual-online-inference
function Rocket.on_next!(listener::MyEventListener, event::RxInferenceEvent{ :after_stop })
    (engine, ) = event
    @test engine isa RxInferenceEngine #hide
    println("The engine has been stopped.")
end
```

```julia
:on_new_data
```
Emits right before processing new data point.
The data is `(model::ProbabilisticModel, data)`

```@example manual-online-inference
function Rocket.on_next!(listener::MyEventListener, event::RxInferenceEvent{ :on_new_data })
    (model, data) = event
    @test model isa ProbabilisticModel #hide
    @test data isa NamedTuple #hide
    @test haskey(data, :y) #hide
    @test iszero(data[:y]) || isone(data[:y]) #hide
    println("The new data point has been received: ", data)
end
```

```julia
:before_iteration
```
Emits right before starting new variational iteration.
The data is `(model::ProbabilisticModel, iteration::Int)`

```@example manual-online-inference
function Rocket.on_next!(listener::MyEventListener, event::RxInferenceEvent{ :before_iteration })
    (model, iteration) = event
    @test model isa ProbabilisticModel #hide
    @test iteration isa Int #hide
    println("Starting new variational iteration #", iteration)
end
```

```julia
:before_auto_update
```
Emits right before executing the [`@autoupdates`](@ref).
The data is `(model::ProbabilisticModel, iteration::Int, autoupdates)`

```@example manual-online-inference
function Rocket.on_next!(listener::MyEventListener, event::RxInferenceEvent{ :before_auto_update })
    (model, iteration, autoupdates) = event
    @test model isa ProbabilisticModel #hide
    @test iteration isa Int #hide
    @test RxInfer.numautoupdates(autoupdates) === 1 #hide
    println("Before processing autoupdates")
end
```

```julia
:after_auto_update
```
Emits right after executing the [`@autoupdates`](@ref).
The data is `(model::ProbabilisticModel, iteration::Int, autoupdates)`

```@example manual-online-inference
function Rocket.on_next!(listener::MyEventListener, event::RxInferenceEvent{ :after_auto_update })
    (model, iteration, autoupdates) = event
    @test model isa ProbabilisticModel #hide
    @test iteration isa Int #hide
    @test RxInfer.numautoupdates(autoupdates) === 1 #hide
    println("After processing autoupdates")
end
```

```julia
:before_data_update
```
Emits right before feeding the model with the new data.
The data is `(model::ProbabilisticModel, iteration::Int, data)`

```@example manual-online-inference
function Rocket.on_next!(listener::MyEventListener, event::RxInferenceEvent{ :before_data_update })
    (model, iteration, data) = event
    @test model isa ProbabilisticModel #hide
    @test iteration isa Int #hide
    println("Before processing new data ", data)
end
```

```julia
:after_data_update
```
Emits right after feeding the model with the new data.
The data is `(model::ProbabilisticModel, iteration::Int, data)`

```@example manual-online-inference
function Rocket.on_next!(listener::MyEventListener, event::RxInferenceEvent{ :after_data_update })
    (model, iteration, data) = event
    @test model isa ProbabilisticModel #hide
    @test iteration isa Int #hide
    println("After processing new data ", data)
end
```

```julia
:after_iteration
```
Emits right after finishing a variational iteration.
The data is `(model::ProbabilisticModel, iteration::Int)`

```@example manual-online-inference
function Rocket.on_next!(listener::MyEventListener, event::RxInferenceEvent{ :after_iteration })
    (model, iteration) = event
    @test model isa ProbabilisticModel #hide
    @test iteration isa Int #hide
    println("Finishing the variational iteration #", iteration)
end
```

```julia
:before_history_save
```
Emits right before saving the history (if requested).
The data is `(model::ProbabilisticModel, )`

```@example manual-online-inference
function Rocket.on_next!(listener::MyEventListener, event::RxInferenceEvent{ :before_history_save })
    (model, ) = event
    @test model isa ProbabilisticModel #hide
    println("Before saving the history")
end
```

```julia
:after_history_save
```
Emits right after saving the history (if requested).
The data is `(model::ProbabilisticModel, )`

```@example manual-online-inference
function Rocket.on_next!(listener::MyEventListener, event::RxInferenceEvent{ :after_history_save })
    (model, ) = event
    @test model isa ProbabilisticModel #hide
    println("After saving the history")
end
```

```julia
:on_tick
```
Emits right after finishing processing the new observations and completing the inference step.
The data is `(model::ProbabilisticModel, )`

```@example manual-online-inference
function Rocket.on_next!(listener::MyEventListener, event::RxInferenceEvent{ :on_tick })
    (model, ) = event
    @test model isa ProbabilisticModel #hide
    println("Finishing the inference for the new observations")
end
```


```julia
:on_error
```
Emits if an error occurs in the inference engine.
The data is `(model::ProbabilisticModel, err::Any)`

```@example manual-online-inference
function Rocket.on_next!(listener::MyEventListener, event::RxInferenceEvent{ :on_error })
    (model, err) = event
    @test model isa ProbabilisticModel #hide
    println("An error occured during the inference procedure: ", err)
end
```

```julia
:on_complete
```
Emits when the `datastream` completes.
The data is `(model::ProbabilisticModel, )`

```@example manual-online-inference
function Rocket.on_next!(listener::MyEventListener, event::RxInferenceEvent{ :on_complete })
    (model, ) = event
    @test model isa ProbabilisticModel #hide
    println("The data stream completed. The inference has been finished.")
end
```

Let's use our event listener with the [`infer`](@ref) function:
```@example manual-online-inference
engine = infer(
    model          = beta_bernoulli_online(),
    datastream     = observations,
    autoupdates    = beta_bernoulli_autoupdates,
    initialization = @initialization(q(θ) = Beta(1, 1)),
    keephistory    = 5,
    iterations     = 2,
    autostart      = false,
    free_energy    = true,
    events         = Val((
        :before_start,
        :after_start,
        :before_stop,
        :after_stop,
        :on_new_data,
        :before_iteration,
        :before_auto_update,
        :after_auto_update,
        :before_data_update,
        :after_data_update,
        :after_iteration,
        :before_history_save,
        :after_history_save,
        :on_tick,
        :on_error,
        :on_complete
    ))
)
```

After we have created the engine, we can subscribe on events and [`RxInfer.start`](@ref) the engine:
```@example manual-online-inference
events_subscription = subscribe!(engine.events, MyEventListener())

RxInfer.start(engine)
nothing #hide
```

The event loop stays idle without new observation and runs again when a new observation becomes available:
```@example manual-online-inference
next!(datastream, rand(rng, distribution))
```

Let's complete the `datastream` 

```@example manual-online-inference
complete!(datastream)
```

In this case, it is not necessary to [`RxInfer.stop`](@ref) the engine, because 
it will be stopped automatically.
```@example manual-online-inference
@test_logs (:warn, r"The engine has been completed.*") RxInfer.stop(engine) #hide
RxInfer.stop(engine)
nothing #hide
```

!!! note
    The `:before_stop` and `:after_stop` events are not emmited in case of the datastream completion. Use the `:on_complete` instead.


## [Using `data` keyword argument with streaming inference](@id manual-online-inference-data)

The streaming version does support static datasets as well. 
Internally, it converts it to a datastream, that emits all observations in a sequntial order without any delay. As an example:

```@example manual-online-inference
staticdata = rand(rng, distribution, 1_000)
```

Use the `data` keyword argument instead of the `datastream` to pass the static data.

```@example manual-online-inference
engine = infer(
    model          = beta_bernoulli_online(),
    data           = (y = staticdata, ),
    autoupdates    = beta_bernoulli_autoupdates,
    initialization = @initialization(q(θ) = Beta(1, 1)),
    keephistory    = 1000,
    autostart      = true,
    free_energy    = true,
)
```

```@example manual-online-inference
engine.history[:θ]
```

```@example manual-online-inference
@gif for posterior in engine.history[:θ]
    rθ = range(0, 1, length = 1000)
    pθ = plot(rθ, (x) -> pdf(posterior, x), fillalpha=0.3, fillrange = 0, label="P(θ|y)", c=3)
    pθ = vline!(pθ, [ hidden_θ ], label = "Real value of θ")

    plot(pθ)
end
```


## [Where to go next?](@id manual-online-inference-where-to-go)

This guide covered some fundamental usages of the [`infer`](@ref) function in the context of streamline inference, 
but did not cover all the available keyword arguments of the function. Read more explanation about the other keyword arguments 
in the [Overview](@ref user-guide-inference-execution) section or check out the [Static Inference](@ref manual-static-inference) section.
Also check out more complex [examples](https://examples.rxinfer.com/).
