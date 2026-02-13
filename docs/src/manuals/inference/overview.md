# [Inference execution](@id user-guide-inference-execution)

The `RxInfer` inference API supports different types of message-passing algorithms (including hybrid algorithms combining several different types). While `RxInfer` implements several algorithms to cater to different computational needs and scenarios, the core message-passing algorithms that form the foundation of our inference capabilities are:

- [Belief Propagation](https://en.wikipedia.org/wiki/Belief_propagation)
- [Variational Message Passing](https://en.wikipedia.org/wiki/Variational_message_passing)

Whereas belief propagation computes exact inference for the random variables of interest, the variational message passing (VMP) is an approximation method that can be applied to a larger range of models.

The inference engine itself isn't aware of different algorithm types and simply does message passing between nodes. However, during the model specification stage user may specify different factorisation constraints around factor nodes with the help of the [`@constraints`](@ref user-guide-constraints-specification) macro. Different factorisation constraints lead to different message passing update rules. See more documentation about constraints specification in the corresponding [section](@ref user-guide-constraints-specification).

## [Automatic inference specification](@id user-guide-inference-execution-automatic-specification)

`RxInfer` exports the [`infer`](@ref) function to quickly run and test your model with both static and asynchronous (real-time) datasets. See more information about the [`infer`](@ref) function on the separate documentation section:

- [Static Inference](@ref manual-static-inference)
- [Streamlined Inference](@ref manual-online-inference)

```@docs
infer
```

#### Note on NamedTuples

When passing `NamedTuple` as a value for some argument, make sure you use a trailing comma for `NamedTuple`s with a single entry. The reason is that Julia treats `returnvars = (x = KeepLast())` and `returnvars = (x = KeepLast(), )` expressions differently. This first expression creates (or **overwrites!**) new local/global variable named `x` with contents `KeepLast()`. The second expression (note trailing comma) creates `NamedTuple` with `x` as a key and `KeepLast()` as a value assigned for this key.

```@example note-named-tuples
using RxInfer #hide
(x = KeepLast()) # defines a variable `x` with the value `KeepLast()`
```

```@example note-named-tuples
(x = KeepLast(), ) # defines a NamedTuple with `x` as one of the keys and value `KeepLast()`
```

- ### `model`

Also read the [Model Specification](@ref user-guide-model-specification) section.

The `model` argument accepts a model specification as its input. The easiest way to create the model is to use the [`@model`](@ref user-guide-model-specification) macro. 
For example:

```@example inference-overview-model-keyword
using RxInfer #hide

@model function beta_bernoulli(y, a, b)
    x  ~ Beta(a, b)
    y .~ Bernoulli(x)
end

result = infer(
    model = beta_bernoulli(a = 1, b = 1),
    data  = (y = [ true, false, false ], )
)

result.posteriors[:x]
```

!!! note
    The `model` keyword argument does not accept a [`ProbabilisticModel`](@ref) instance as a value, as it needs to inject `constraints` and `meta` during the inference procedure.

- ### `data`

Either `data` or `datastream` keyword argument are required. 
Specifying both `data` and `datastream` is not supported and will result in an error. 

!!! note 
    The behavior of the `data` keyword argument depends on the inference setting ([batch](@ref manual-static-inference) or [streamline](@ref manual-online-inference)).

The `data` keyword argument must be a `NamedTuple` (or `Dict`) where keys (of `Symbol` type) correspond to some arguments defined in the model specification. 
For example, if a model defines `y` in its argument list 
```@example inference-overview-data-keyword
using RxInfer #hide
@model function beta_bernoulli(y, a, b)
    x  ~ Beta(a, b)
    y .~ Bernoulli(x)
end
```
and you want to condition on this argument, then the `data` field must have an `:y` key (of `Symbol` type) which holds the data. 
The values in the `data` must have the exact same shape as its corresponding variable container. E.g. in the exampl above `y` is being used in the broadcasting 
operation, thus it must be a collection of values. `a` and `b` arguments, however, could be just single numbers:
```@example inference-overview-data-keyword
result = infer(
    model = beta_bernoulli(),
    data  = (y = [ true, false, false ], a = 1, b = 1)
)

result.posteriors[:x]
```

- ### `datastream`

Also read the [Streamlined Inference](@ref manual-online-inference) section.

The `datastream` keyword argument must be an observable that supports `subscribe!` and `unsubscribe!` functions (e.g., streams from the `Rocket.jl` package).
The elements of the observable must be of type `NamedTuple` where keys (of `Symbol` type) correspond to input arguments defined in the model specification, except for those which are listed in the [`@autoupdates`](@ref) specification. 
For example, if a model defines `y` as its argument (which is not part of the [`@autoupdates`](@ref) specification) the named tuple from the observable must have an `:y` key (of `Symbol` type). The values in the named tuple must have the exact same shape as the corresponding variable container.

- ### `initialization`

Also read the [Initialization](@ref initialization) section.

For specific types of inference algorithms, such as variational message passing, it might be required to initialize (some of) the marginals before running the inference procedure in order to break the dependency loop. If this is not done, the inference algorithm will not be executed due to the lack of information and message and/or marginals will not be updated. In order to specify these initial marginals and messages, you can use the `initialization` argument in combination with the [`@initialization`](@ref) macro, such as
```@example inference-overview-init-keyword
using RxInfer #hide
init = @initialization begin
    # initialize the marginal distribution of x as a vague Normal distribution
    # if x is a vector, then it simply uses the same value for all elements
    # However, it is also possible to provide a vector of distributions to set each element individually 
    q(x) = vague(NormalMeanPrecision)
end
```

- ### `returnvars`

`returnvars` specifies latent variables of interest and their posterior updates. Its behavior depends on the inference type: streamline or batch.

**Batch inference:**

- Accepts a `NamedTuple` or `Dict` of return variable specifications.
- Two specifications available: `KeepLast` (saves the last update) and `KeepEach` (saves all updates).
- When `iterations` is set, returns every update for each iteration (equivalent to `KeepEach()`); if `nothing`, saves the last update (equivalent to `KeepLast()`).
- Use `iterations = 1` to force `KeepEach()` for a single iteration or set `returnvars = KeepEach()` manually.

```julia
result = infer(
    ...,
    returnvars = (
        x = KeepLast(),
        τ = KeepEach()
    )
)
```

Shortcut for setting the same option for all variables:

```julia
result = infer(
    ...,
    returnvars = KeepLast()  # or KeepEach()
)
```

**Streamline inference:**

- For each symbol in `returnvars`, `infer` creates an observable stream of posterior updates.
- Agents can subscribe to these updates using the `Rocket.jl` package.

```julia
engine = infer(
    ...,
    autoupdates = my_autoupdates,
    returnvars = (:x, :τ),
    autostart  = false
)
```

```@docs
KeepLast
KeepEach
```

- ### `predictvars`

`predictvars` specifies the variables which should be predicted.
Similar to `returnvars`, `predictvars` accepts a `NamedTuple` or `Dict`. There are two specifications:
- `KeepLast`: saves the last update for a variable, ignoring any intermediate results during iterations
- `KeepEach`: saves all updates for a variable for all iterations

```julia
result = infer(
    ...,
    predictvars = (
        o = KeepLast(),
        τ = KeepEach()
    )
)
```

!!! note
    The `predictvars` argument is exclusive for batch setting.

- ### `historyvars`

Also read the [Keeping the history of posteriors](@ref manual-online-inference-history).

`historyvars` specifies the variables of interests and the amount of information to keep in history about the posterior updates when performing streamline inference. The specification is similar to the `returnvars` when applied in batch setting.
The `historyvars` requires `keephistory` to be greater than zero.

`historyvars` accepts a `NamedTuple` or `Dict` or return var specification. There are two specifications:
- `KeepLast`: saves the last update for a variable, ignoring any intermediate results during iterations
- `KeepEach`: saves all updates for a variable for all iterations

```julia
result = infer(
    ...,
    autoupdates = my_autoupdates,
    historyvars = (
        x = KeepLast(),
        τ = KeepEach()
    ),
    keephistory = 10
)
```

It is also possible to set either `historyvars = KeepLast()` or `historyvars = KeepEach()` that acts as an alias and sets the given option for __all__ random variables in the model.

```julia
result = infer(
    ...,
    autoupdates = my_autoupdates,
    historyvars = KeepLast(),
    keephistory = 10
)
```

- ### `keep_history`

Specifies the buffer size for the updates history both for the `historyvars` and the `free_energy` buffers in streamline inference.

!!! note
    The `historyvars` and `keep_history` arguments are exclusive for streamlined setting.

- ### `iterations`

Specifies the number of variational (or loopy belief propagation) iterations. By default set to `nothing`, which is equivalent of doing 1 iteration. However, if set explicitly to `1` the default setting for `returnvars` changes from `KeepLast` to `KeepEach`.

See [Early stopping](@ref manual-inference-early-stopping) for an opt-in callback example that implements early stopping criterion.

- ### `free_energy`

**Batch inference:**

Specifies if the `infer` function should return Bethe Free Energy (BFE) values.

- Optionally accepts a floating-point type (e.g., `Float64`) for improved BFE computation performance, but restricts the use of automatic differentiation packages.

**Streamline inference:**

Specifies if the `infer` function should create an observable stream of Bethe Free Energy (BFE) values, computed at each VMP iteration.

- When `free_energy = true` and `keephistory > 0`, additional fields are exposed in the engine for accessing the history of BFE updates.
  - `engine.free_energy_history`: Averaged BFE history over VMP iterations.
  - `engine.free_energy_final_only_history`: BFE history of values computed in the last VMP iterations for each observation.
  - `engine.free_energy_raw_history`: Raw BFE history.


- ### `free_energy_diagnostics`

This settings specifies either a single or a tuple of diagnostic checks for Bethe Free Energy values stream. By default checks for `NaN`s and `Inf`s. 
See also [`RxInfer.ObjectiveDiagnosticCheckNaNs`](@ref) and [`RxInfer.ObjectiveDiagnosticCheckInfs`](@ref).
Pass `nothing` to disable any checks.

- ### `options`

```@docs
RxInfer.ReactiveMPInferenceOptions
```

- ### `catch_exception`

The `catch_exception` keyword argument specifies whether exceptions during the batch inference procedure should be caught in the `error` field of the 
result. By default, if exception occurs during the inference procedure the result will be lost. Set `catch_exception = true` to obtain partial result 
for the inference in case if an exception occurs. Use [`RxInfer.issuccess`](@ref) and [`RxInfer.iserror`](@ref) function to check if the inference completed successfully or failed.
If an error occurs, the `error` field will store a tuple, where first element is the exception itself and the second element is the caught `backtrace`. Use the `stacktrace` function 
with the `backtrace` as an argument to recover the stacktrace of the error. Use `Base.showerror` function to display
the error.

```@docs
RxInfer.issuccess
RxInfer.iserror
```

- ### `callbacks`

The inference function has its own lifecycle. The user is free to provide some (or none) of the callbacks to inject some extra logging or other procedures in the inference function, e.g.

```julia
result = infer(
    ...,
    callbacks = (
        on_marginal_update = (model, name, update) -> println("\$(name) has been updated: \$(update)"),
        after_inference    = (args...) -> println("Inference has been completed")
    )
)
```

The `callbacks` keyword argument accepts a named-tuple of 'name = callback' pairs. 
The list of all possible callbacks for different inference setting (batch or streamline) and their arguments is present below:

- `before_model_creation()`
- `after_model_creation(model::ProbabilisticModel)`

**Exlusive for batch inference**

- `on_marginal_update(model::ProbabilisticModel, name::Symbol, update)`
- `before_inference(model::ProbabilisticModel)`
- `before_iteration(model::ProbabilisticModel, iteration::Int)::Bool`
- `before_data_update(model::ProbabilisticModel, data)`
- `after_data_update(model::ProbabilisticModel, data)`
- `after_iteration(model::ProbabilisticModel, iteration::Int)::Bool`
- `after_inference(model::ProbabilisticModel)`

!!! note
    `before_iteration` and `after_iteration` callbacks are allowed to return `true/false` value. `true` indicates that iterations must be halted and no further inference should be made.

**Exlusive for streamline inference**

- `before_autostart(engine::RxInferenceEngine)`
- `after_autostart(engine::RxInferenceEngine)`

See [Early stopping](@ref manual-inference-early-stopping) for an opt-in callback example, which implements early stopping criterion for Free Energy.
See [Benchmarking RxInfer via callbacks](@ref user-guide-debugging-benchmark-callbacks) for another examples demonstrating how to benchmark inference via callbacks.

- ### `addons`

The `addons` field extends the default message computation rules with some extra information, e.g. computing log-scaling factors of messages or saving debug-information. Accepts a single addon or a tuple of addons. 
Automatically changes the default value of the `postprocess` argument to `NoopPostprocess`.

- ### `postprocess`

Also read the [Inference results postprocessing](@ref user-guide-inference-postprocess) section.

The `postprocess` keyword argument controls whether the inference results must be modified in some way before exiting the `inference` function.
By default, the inference function uses the `DefaultPostprocess` strategy, which by default removes the `Marginal` wrapper type from the results.
Change this setting to `NoopPostprocess` if you would like to keep the `Marginal` wrapper type, which might be useful in the combination with the `addons` argument.
If the `addons` argument has been used, automatically changes the default strategy value to `NoopPostprocess`.

- ### Error hints

By default, RxInfer provides helpful error hints when an error occurs during inference.
This, for example, includes links to relevant documentation, common solutions and troubleshooting steps, information about where to get help, and suggestions for providing good bug reports.

Use [`RxInfer.disable_inference_error_hint!`](@ref) to disable error hints or [`RxInfer.enable_inference_error_hint!`](@ref) to enable them. Note that the change requires a Julia session restart to take effect.

```@docs
RxInfer.disable_inference_error_hint!
RxInfer.enable_inference_error_hint!
```

## Where to go next?

Read more explanation about the other keyword arguments in the [Streamlined (online) inference](@ref manual-online-inference)section or check out the [Static Inference](@ref manual-static-inference) section or check some more advanced [examples](https://examples.rxinfer.com/).
