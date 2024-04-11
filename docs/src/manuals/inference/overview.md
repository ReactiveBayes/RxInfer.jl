# [Inference execution](@id user-guide-inference-execution)

The `RxInfer` inference API supports different types of message-passing algorithms (including hybrid algorithms combining several different types):

- [Belief Propagation](https://en.wikipedia.org/wiki/Belief_propagation)
- [Variational Message Passing](https://en.wikipedia.org/wiki/Variational_message_passing)

Whereas belief propagation computes exact inference for the random variables of interest, the variational message passing (VMP) in an approximation method that can be applied to a larger range of models.

The inference engine itself isn't aware of different algorithm types and simply does message passing between nodes, however during model specification stage user may specify different factorisation constraints around factor nodes with the help of the [`@constraints`](@ref user-guide-constraints-specification) macro. Different factorisation constraints lead to a different message passing update rules. See more documentation about constraints specification in the corresponding [section](@ref user-guide-constraints-specification).

## [Automatic inference specification](@id user-guide-inference-execution-automatic-specification)

`RxInfer` exports the [`infer`](@ref) function to quickly run and test you model with both static and asynchronous (real-time) datasets. See more information about the `infer` function on the separate documentation section:

- [Static Inference](@ref manual-static-inference). 
- [Streamlined Inference](@ref manual-online-inference). 

```@docs
infer
RxInfer.ReactiveMPInferenceOptions
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

Either `data` or `datastream` or `predictvars` keyword argument is required. Specifying both `data` and `datastream` is not supported and will result in an error. Specifying both `datastream` and `predictvars` is not supported and will result in an error.

**Note**: The behavior of the `data` keyword argument depends on the inference setting (batch or streamline).

The `data` keyword argument must be a `NamedTuple` (or `Dict`) where keys (of `Symbol` type) correspond to all `datavar`s defined in the model specification. For example, if a model defines `x = datavar(Float64)` the 
`data` field must have an `:x` key (of `Symbol` type) which holds a value of type `Float64`. The values in the `data` must have the exact same shape as the `datavar` container. In other words, if a model defines `x = datavar(Float64, n)` then 
`data[:x]` must provide a container with length `n` and with elements of type `Float64`.

#### `streamline` setting

All entries in the `data` argument are zipped together with the `Base.zip` function to form one slice of the data chunck. This means all containers in the `data` argument must be of the same size (`zip` iterator finished as soon as one container has no remaining values).
In order to use a fixed value for some specific `datavar` it is not necessary to create a container with that fixed value, but rather more efficient to use `Iterators.repeated` to create an infinite iterator.

- ### `datastream`

The `datastream` keyword argument must be an observable that supports `subscribe!` and `unsubscribe!` functions (streams from the `Rocket.jl` package are also supported).
The elements of the observable must be of type `NamedTuple` where keys (of `Symbol` type) correspond to all `datavar`s defined in the model specification, except for those which are listed in the `autoupdates` specification. 
For example, if a model defines `x = datavar(Float64)` (which is not part of the `autoupdates` specification) the named tuple from the observable must have an `:x` key (of `Symbol` type) which holds a value of type `Float64`. The values in the named tuple must have the exact same shape as the `datavar` container. In other words, if a model defines `x = datavar(Float64, n)` then 
`namedtuple[:x]` must provide a container with length `n` and with elements of type `Float64`.

**Note**: The behavior of the individual named tuples from the `datastream` observable is similar to that which is used in the batch setting.
In fact, you can see the streamline inference as an efficient version of the batch inference, which automatically updates some `datavar`s with the `autoupdates` specification and listens to the `datastream` to update the rest of the `datavar`s.

- ### `initialization`

For specific types of inference algorithms, such as variational message passing, it might be required to initialize (some of) the marginals before running the inference procedure in order to break the dependency loop. If this is not done, the inference algorithm will not be executed due to the lack of information and message and/or marginals will not be updated. In order to specify these initial marginals and messages, you can use the `initialization` argument in combination with the `@initialization` macro, such as
```julia
init = @initialization begin
    # initialize the marginal distribution of x as a vague Normal distribution
    # if x is a vector, then it simply uses the same value for all elements
    # However, it is also possible to provide a vector of distributions to set each element individually 
    q(x) = vague(NormalMeanPrecision)
end

infer(...
    initialization = init,
)

This argument needs to be a named tuple, i.e. `initmarginals = (a = ..., )`, or dictionary.

- ### `initmessages`

Depracated initialization for messages, use `initialization` instead

- ### `initmarginals`

Depracated initialization for marginals, use `initialization` instead

- ### `options`

- `limit_stack_depth`: limits the stack depth for computing messages, helps with `StackOverflowError` for some huge models, but reduces the performance of inference backend. Accepts integer as an argument that specifies the maximum number of recursive depth. Lower is better for stack overflow error, but worse for performance.
- `pipeline`: changes the default pipeline for each factor node in the graph
- `global_reactive_scheduler`: changes the scheduler of reactive streams, see Rocket.jl for more info, defaults to no scheduler

- ### `returnvars`

`returnvars` specifies latent variables of interest and their posterior updates. Its behavior depends on the inference type: streamline or batch.

**Batch inference:**
- Accepts a `NamedTuple` or `Dict` of return variable specifications.
- Two specifications available: `KeepLast` (saves the last update) and `KeepEach` (saves all updates).
- When `iterations` is set, returns every update for each iteration (equivalent to `KeepEach()`); if `nothing`, saves the last update (equivalent to `KeepLast()`).
- Use `iterations = 1` to force `KeepEach()` for a single iteration or set `returnvars = KeepEach()` manually.

Example:

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

Example:

```julia
engine = infer(
    ...,
    autoupdates = my_autoupdates,
    returnvars = (:x, :τ),
    autostart  = false
)
```

- ### `predictvars`

`predictvars` specifies the variables which should be predicted. In the model definition these variables are specified
as datavars, although they should not be passed inside data argument.

Similar to `returnvars`, `predictvars` accepts a `NamedTuple` or `Dict`. There are two specifications:
- `KeepLast`: saves the last update for a variable, ignoring any intermediate results during iterations
- `KeepEach`: saves all updates for a variable for all iterations

Example: 

```julia
result = infer(
    ...,
    predictvars = (
        o = KeepLast(),
        τ = KeepEach()
    )
)
```

**Note**: The `predictvars` argument is exclusive for batch setting.

- ### `historyvars`

`historyvars` specifies the variables of interests and the amount of information to keep in history about the posterior updates when performing streamline inference. The specification is similar to the `returnvars` when applied in batch setting.
The `historyvars` requires `keephistory` to be greater than zero.

`historyvars` accepts a `NamedTuple` or `Dict` or return var specification. There are two specifications:
- `KeepLast`: saves the last update for a variable, ignoring any intermediate results during iterations
- `KeepEach`: saves all updates for a variable for all iterations

Example: 

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

# Example: 

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

- ### `iterations`

Specifies the number of variational (or loopy belief propagation) iterations. By default set to `nothing`, which is equivalent of doing 1 iteration. 

- ### `free_energy`

**Streamline inference:**

Specifies if the `infer` function should create an observable stream of Bethe Free Energy (BFE) values, computed at each VMP iteration.

- When `free_energy = true` and `keephistory > 0`, additional fields are exposed in the engine for accessing the history of BFE updates.
  - `engine.free_energy_history`: Averaged BFE history over VMP iterations.
  - `engine.free_energy_final_only_history`: BFE history of values computed in the last VMP iterations for each observation.
  - `engine.free_energy_raw_history`: Raw BFE history.

**Batch inference:**

Specifies if the `infer` function should return Bethe Free Energy (BFE) values.

- Optionally accepts a floating-point type (e.g., `Float64`) for improved BFE computation performance, but restricts the use of automatic differentiation packages.

- ### `free_energy_diagnostics`

This settings specifies either a single or a tuple of diagnostic checks for Bethe Free Energy values stream. By default checks for `NaN`s and `Inf`s. 
See also [`RxInfer.ObjectiveDiagnosticCheckNaNs`](@ref) and [`RxInfer.ObjectiveDiagnosticCheckInfs`](@ref).
Pass `nothing` to disable any checks.

- ### `catch_exception`

The `catch_exception` keyword argument specifies whether exceptions during the batch inference procedure should be caught in the `error` field of the 
result. By default, if exception occurs during the inference procedure the result will be lost. Set `catch_exception = true` to obtain partial result 
for the inference in case if an exception occurs. Use `RxInfer.issuccess` and `RxInfer.iserror` function to check if the inference completed successfully or failed.
If an error occurs, the `error` field will store a tuple, where first element is the exception itself and the second element is the caught `backtrace`. Use the `stacktrace` function 
with the `backtrace` as an argument to recover the stacktrace of the error. Use `Base.showerror` function to display
the error.

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

- `on_marginal_update`:    args: (model::FactorGraphModel, name::Symbol, update) (exlusive for batch inference)
- `before_model_creation`: args: ()
- `after_model_creation`:  args: (model::FactorGraphModel)
- `before_inference`:      args: (model::FactorGraphModel) (exlusive for batch inference)
- `before_iteration`:      args: (model::FactorGraphModel, iteration::Int)::Bool (exlusive for batch inference)
- `before_data_update`:    args: (model::FactorGraphModel, data) (exlusive for batch inference)
- `after_data_update`:     args: (model::FactorGraphModel, data) (exlusive for batch inference)
- `after_iteration`:       args: (model::FactorGraphModel, iteration::Int)::Bool (exlusive for batch inference)
- `after_inference`:       args: (model::FactorGraphModel) (exlusive for batch inference)
- `before_autostart`:      args: (engine::RxInferenceEngine) (exlusive for streamline inference)
- `after_autostart`:       args: (engine::RxInferenceEngine) (exlusive for streamline inference)

`before_iteration` and `after_iteration` callbacks are allowed to return `true/false` value.
`true` indicates that iterations must be halted and no further inference should be made.

- ### `addons`

The `addons` field extends the default message computation rules with some extra information, e.g. computing log-scaling factors of messages or saving debug-information.
Accepts a single addon or a tuple of addons. If set, replaces the corresponding setting in the `options`. Automatically changes the default value of the `postprocess` argument to `NoopPostprocess`.

- ### `postprocess`

The `postprocess` keyword argument controls whether the inference results must be modified in some way before exiting the `inference` function.
By default, the inference function uses the `DefaultPostprocess` strategy, which by default removes the `Marginal` wrapper type from the results.
Change this setting to `NoopPostprocess` if you would like to keep the `Marginal` wrapper type, which might be useful in the combination with the `addons` argument.
If the `addons` argument has been used, automatically changes the default strategy value to `NoopPostprocess`.
