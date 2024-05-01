# [Model Specification](@id user-guide-model-specification)

`RxInfer` largely depends on `GraphPPL` for model specification. Read extensive documentation regarding the model specification in the corresponding section of [`GraphPPL` documentation](https://reactivebayes.github.io/GraphPPL.jl/stable/). Here we outline only a small portion of model specification capabilities for beginners.

## `@model` macro

The `RxInfer.jl` package exports the `@model` macro for model specification. This `@model` macro accepts the model specification itself in a form of regular Julia function. 

For example: 
```julia
@model function model_name(model_arguments...)
    # model specification here
end
```
where `model_arguments...` may include both hypeparameters and data. 

!!! note
    `model_arguments` are converted to keyword arguments. Positional arguments in the model specification are not supported. 
    Thus it is not possible to use Julia's multiple dispatch for the model arguments.

The `@model` macro returns a regular Julia function (in this example `model_name()`) which can be executed as usual. The only difference here is that all arguments of the model function are treated as keyword arguments. Upon calling, the model function returns a so-called model generator object, e.g:

```@example model-specification-model-macro
using RxInfer #hide
@model function my_model(observation, hyperparameter)
    observations ~ Normal(0.0, hyperparameter)
end
```

```@example model-specification-model-macro
model = my_model(hyperparameter = 3)
nothing #hide
```

The model generator is not a real model (yet). For example, in the code above, we haven't specified anything for the `observation`. 
The generator object allows us to iteratively add extra properties to the model, condition on data, and/or assign extra metadata information without actually materializing the entire graph structure. Read extra information about model generator [here](@ref lib-model-construction).

## A state space model example

Here we give an example of a probabilistic model before presenting the details of the model specification syntax.
The model below is a simple state space model with latent random variables `x` and noisy observations `y`.

```@example model-specification-ssm
using RxInfer #hide

@model function state_space_model(y, trend, variance)
    x[1] ~ Normal(mean = 0.0, variance = 100.0)
    y[1] ~ Normal(mean = x[1], variance = variance)
    for i in 2:length(y)
       x[i] ~ Normal(mean = x[i - 1] + trend, variance = 1.0)
       y[i] ~ Normal(mean = x[i], variance = variance)
    end
end
```

In this model we assign a prior distribution over latent state `x[1]`. All subsequent states `x[i]` depend on `x[i - 1]` and `trend` and are modelled 
as a simple [Gaussian random walk](https://en.wikipedia.org/wiki/Random_walk#:~:text=Gaussian%20random%20walk,-A%20random%20walk&text=If%20%CE%BC%20is%20nonzero%2C%20the,will%20be%20vs%20%2B%20n%CE%BC.). Observations `y` are modelled with the `Gaussian` distribution as well with a 
prespecified `variance` hyperparameter.

!!! note
    `length(y)` can be called only if `y` has an associated data with it. This is not always the case, for example it is possible to instantiate the 
    model lazily before the data becomes available. In such situations, `length(y)` will throw an error.

### [Hyperparameters](@id user-guide-model-specification-hyperparameters)

Any constant passed to a model as a model argument will be automatically converted to a corresponding constant node in the model's graph.

```@example model-specification-ssm
model = state_space_model(trend = 3.0, variance = 1.0)
nothing #hide
```

In this example we instantiate a model generator with `trend` and `variance` parameters _clamped_ to `3.0` and `1.0` respectively. That means 
that no inference will be performed for those parameters and some of the expressions within the model structure might be simplified and compiled-out.

### [Conditioning on data](@id user-guide-model-specification-conditioning)

To fully complete model specification we need to specify `y`. In this example, `y` is playing a role of observations.
`RxInfer` provides a convenient mechanism to pass data values to the model with the `|` operator.

```@example model-specification-ssm
conditioned = model | (y = [ 0.0, 1.0, 2.0 ], )
```

!!! note
    The conditioning on data is a feature of `RxInfer`, not `GraphPPL`.

In the example above we conditioned on data in a form of the `NamedTuple`, but it is also possible to 
condition on a dictionary where keys represent names of the corresponding model arguments:
```@example model-specification-ssm
data        = Dict(:y => [ 0.0, 1.0, 2.0 ])
conditioned = model | data
```

Sometimes it might be useful to indicate that some arguments are data (thus condition on them) before the actual data becomes available.
This situation may occur during [reactive inference](@ref manual-online-inference), when data becomes available _after_ model creation.
`RxInfer` provides a special structure called [`RxInfer.DeferredDataHandler`](@ref), which can be used instead of the real data.

For the example above, however, we cannot simply do the following:
```julia
model | (y = RxInfer.DeferredDataHandler(), )
```
because we use `length(y)` in the model and this is only possible if `y` has an associated data. 
We could adjust the model specification a bit, by adding the extra `n` parameter to the list of arguments:
```@example model-specification-ssm
@model function state_space_model_with_n(y, n, trend, variance)
    x[1] ~ Normal(mean = 0.0, variance = 100.0)
    y[1] ~ Normal(mean = x[1], variance = variance)
    for i in 2:n
       x[i] ~ Normal(mean = x[i - 1] + trend, variance = 1.0)
       y[i] ~ Normal(mean = x[i], variance = variance)
    end
end
```

For such model, we can safely condition on `y` without providing actual data for it, but using the [`RxInfer.DeferredDataHandler`](@ref) instead:
```@example model-specification-ssm
state_space_model_with_n(trend = 3.0, variance = 1.0, n = 10) | (
    y = RxInfer.DeferredDataHandler(), 
)
```

Read more information about condition on data in [this section](@ref lib-model-construction-conditioning) of the documentation.

### [Latent variables](@id user-guide-model-specification-random-variables)

Latent variables are being created with the `~` operator and can be read as _is distributed as_. 
For example, to create a latent variable `y` which is modeled by a [Normal](https://en.wikipedia.org/wiki/Normal_distribution) distribution, 
where its mean and variance are controlled by the random variables `m` and `v` respectively, we define

```julia
y ~ Normal(mean = m, variance = v)
```

In the example above
```julia
x[1] ~ Normal(mean = 0.0, variance = 100.0)
```
indicates that `x₁` is distributed as [Normal](https://en.wikipedia.org/wiki/Normal_distribution) distribution. 

!!! note
    The `RxInfer.jl` package uses the `~` operator for modelling both stochastic and deterministic relationships between random variables.
    However, `GraphPPL.jl` also allows to use `:=` operator for deterministic relationships.

## [Relationships between variables](@id user-guide-model-specification-node-creation)

In probabilistic models based on graphs, factor nodes are used to define a relationship between random variables and/or constants and data variables.
A factor node defines a probability distribution over selected latent or data variables. The `~` operator not only creates a latent variable but also 
defines a functional relatinship of it with other variables and creates a factor node as a result.

In the example above
```julia
x[1] ~ Normal(mean = 0.0, variance = 100.0)
```
not only creates a latent variable `x₁` but also a factor node `Normal`.

!!! note
    Generally it is not necessary to label all the arguments with their names, as `mean = ...` or `variance = ...` and many factor nodes 
    do not require it explicitly. However, for nodes, which have many different useful parametrizations (e.g. `Normal`) labeling the arguments 
    is a requirement that helps to avoid any possible confusion. Read more about `Distributions` compatibility [here](@ref user-guide-model-specification-distributions).

### [Control flow statements](@id user-guide-model-specification-node-creation-control-flow)

In general, it is possible to use any Julia code within model specification function, including control flow statements, such as `for`, `while` and `if` statements. However, it is not possible to use any latent states within such statements. This is due to the fact that it is necessary to know exactly the structure of the graph before the inference. Thus it is **not possible** to write statements like:
```julia
c ~ Categorical([ 1/2, 1/2 ])
# This is NOT possible in `RxInfer`'s model specification language
if c > 1
# ...
end
```
since `c` must be statically known upon graph creation.

### [Anonymous factor nodes and latent variables](@id user-guide-model-specification-node-creation-anonymous)

The `@model` macro automatically resolves any inner function calls into anonymous factor nodes and latent variables. 
For example the following:
```julia
y ~ Normal(
    mean = Normal(mean = 0.0, variance = 1.0), 
    precision = Gamma(shape = 1.0, rate = 1.0)
)
```
is equivalent to
```julia
tmp1 ~ Normal(mean = 0.0, variance = 1.0)
tmp2 ~ Gamma(shape = 1.0, rate = 1.0)
y    ~ Normal(mean = tmp1, precision = tmp2)
```

The inference backend still performs inference for anonymous latent variables, however, there it does not provide an easy way to obtain posteriors for them.
Note that the inference backend will try to optimize deterministic function calls in the case where all arguments are known in advance.
For example:
```julia
y ~ Normal(mean = 0.0, variance = inv(2.0))
```
should not create an extra factor node for the `inv`, since `inv` is a deterministic function and all arguments are known in advance. The same situation
applies in case of complex initializations involving different types, as in:
```julia
y ~ MvNormal(mean = zeros(3), covariance = Matrix(Diagonal(ones(3))))
```
In this case, the expression `Matrix(Diagonal(ones(3)))` can (and will) be precomputed upon model creation and does not require to perform probabilistic inference.

### [Indexing operations](@id user-guide-model-specification-node-creation-indexing)

The `ref` expressions, such as `x[i]`, are handled in a special way.
Technically, in Julia, the `x[i]` call is translated to a function call `getindex(x, i)`. Thus the `@model` macro should create a factor node for the `getindex` function, but this won't happen in practice because this case is treated separately. This means that the model parser will not create unnecessary nodes when only simple indexing is involved. That also means that all expressions inside `x[...]` list are left untouched during model parsing. 

!!! warning
    It is not allowed to use latent variables within square brackets in the model specification or for control flow statements such as `if`, `for` or `while`.

### [Broadcasting syntax](@id user-guide-model-specification-node-creation-broadcasting)

`GraphPPL` support broadcasting for `~` operator in the exact same way as Julia itself. 
A user is free to write an expression of the following form:
```julia
m  ~ Normal(mean = 0.0, precision = 0.0001)
t  ~ Gamma(shape = 1.0, rate = 1.0)
y .~ Normal(mean = m, precision = t)
```

More complex expressions are also allowed:
```julia
w         ~ Wishart(3, diageye(2))
x[1]      ~ MvNormal(mean = zeros(2), precision = diageye(2))
x[2:end] .~ A .* x[1:end-1] # <- State-space model with transition matrix A
y        .~ MvNormal(mean = x, precision = w) # <- Observations with unknown precision matrix
```

Note, however, that shapes of all variables that take part in the broadcasting operation must be defined in advance. That means that it is not possible to 
use broadcasting with [deffered data](@ref user-guide-model-specification-conditioning). Read more about how broadcasting machinery works in Julia in [the official documentation](https://docs.julialang.org/en/v1/manual/arrays/#Broadcasting).

### [`Distributions.jl` compatibility](@id user-guide-model-specification-distributions)

For some factor nodes we rely on the syntax from `Distributions.jl` to make it easy to adopt `RxInfer.jl` for these users. These nodes include for example the [`Beta`](https://en.wikipedia.org/wiki/Beta_distribution) and [`Wishart`](https://en.wikipedia.org/wiki/Wishart_distribution) distributions. These nodes can be created using the `~` syntax with the arguments as specified in `Distributions.jl`. Unfortunately, we `RxInfer.jl` is not yet compatible with all possible distributions to be used as factor nodes. If you feel that you would like to see another node implemented, please file an issue.

!!! note
    To quickly check the list of all available factor nodes that can be used in the model specification language call `?ReactiveMP.is_predefined_node` or `Base.doc(ReactiveMP.is_predefined_node)`.


Specifically for the Gaussian/Normal case we have custom implementations that yield a higher computational efficiency and improved stability in comparison to `Distributions.jl` as these are optimized for sampling operations. Our aliases for these distributions therefore do not correspond to the implementations from `Distributions.jl`. However, our model specification language is compatible with syntax from `Distributions.jl` for normal distributions, which will be automatically converted. `RxInfer` has its own implementation because of the following 3 reasons:
1. `Distributions.jl` constructs normal distributions by saving the corresponding covariance matrices in a `PDMat` object from `PDMats.jl`. This construction always computes the Cholesky decompositions of the covariance matrices, which is very convenient for sampling-based procedures. However, in `RxInfer.jl` we mostly base our computations on analytical expressions which do not always need to compute the Cholesky decomposition. In order to reduce the overhead that `Distributions.jl` introduces, we therefore have custom implementations.
2. Depending on the update rules, we might favor different parameterizations of the normal distributions. `ReactiveMP.jl` has quite a variety in parameterizations that allow us to efficient computations where we convert between parameterizations as little as possible.
3. In certain situations we value stability a lot, especially when inverting matrices. `PDMats.jl`, and hence `Distributions.jl`, is not capable to fulfill all needs that we have here. Therefore we use `PositiveFactorizations.jl` to cope with the corner-cases.

## [Model structure visualisation](@id user-guide-model-specification-visualization)

It is also possible to visualize the model structure after conditioning on data. For that we need two extra packages installed: `Cairo` and `GraphPlot`. Note, that those packages are not included in the `RxInfer` package and must be installed separately.

```@example model-specification-ssm
using Cairo, GraphPlot

# `Create` the actual graph of the model conditioned on the data
model = RxInfer.create_model(conditioned)

# Call `gplot` function from `GraphPlot` to visualise the structure of the graph
GraphPlot.gplot(RxInfer.getmodel(model))
```

### [Node creation options](@id user-guide-model-specification-node-creation-options)

`GraphPPL` allows to pass optional arguments to the node creation constructor with the `where { options...  }` options specification syntax.

Example:
```julia
y ~ Normal(mean = y_mean, var = y_var) where { meta = ... }
```

A list of the available options specific to the `ReactiveMP` inference engine is presented below.

#### Metadata option

Is is possible to pass any extra metadata to a factor node with the `meta` option. Metadata can be later accessed in message computation rules.
```julia
z ~ f(x, y) where { meta = Linearization() }
d ~ g(a, b) where { meta = Unscented() }
```
This option might be useful to change message passing rules around a specific factor node. Read more about this feature in [Meta Specification](@ref user-guide-meta-specification) section.

#### Dependencies option

A user can modify default computational pipeline of a node with the `dependencies` options. 
Read more about different options in the [`ReactiveMP.jl` documentation](https://reactivebayes.github.io/ReactiveMP.jl/stable/).

```julia
y[k - 1] ~ Probit(x[k]) where {
    # This specification indicates that in order to compute an outbound message from the `in` interface
    # We need an inbound message from the same edge initialized to `NormalMeanPrecision(0.0, 1.0)`
    dependencies = RequireMessageFunctionalDependencies(in = NormalMeanPrecision(0.0, 1.0))
}
```

## Read also

- [Constraints specification](@ref user-guide-constraints-specification)
- [Meta specification](@ref user-guide-meta-specification)
- [Inference execution](@ref user-guide-inference-execution)
- [Debugging inference](@ref user-guide-debugging)
