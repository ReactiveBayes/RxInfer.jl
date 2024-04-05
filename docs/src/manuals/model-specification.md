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

The `@model` macro returns a regular Julia function (in this example `model_name()`) which can be executed as usual. It returns a so-called model generator object, e.g:

```@example model-specification-model-macro
@model function my_model(observation, hyperparameter)
    observations ~ Normal(0.0, hyperparameter)
end
```

```@example model-specification-model-macro
model = my_model(hyperparameter = 3)
#nothing hide
```

## A state space model example

Here we give an example of a probabilistic model is given before presenting the details of the model specification syntax.
The model below is a simple state space model with latent random variables `x` and noisy observations `y`.

```@example model-specification-ssm
@model function state_space_model(y, trend, variance)
    x[1] ~ Normal(mean = 0.0, variance = 100.0)
    for i in 2:length(y)
       x[i] ~ Normal(mean = x[i - 1] + trend, variance = 1.0)
       y[i] ~ Normal(mean = x[i], variance = variance)
    end
end
```

### [Hyper-parameters](@id user-guide-model-specification-hyperparameters)

Any constant passed to a model as a model argument will be automatically converted to a corresponding constant node in the model's graph.

```@example model-specification-ssm
model = state_space_model(trend = 3.0, variance = 1.0)
nothing #hide
```

### [Conditioning on data](@id user-guide-model-specification-conditioning)

It is important to have a mechanism to pass data values to the model. We can condition on the available data with the `|` operator.

```@example model-specification-ssm
conditioned = model | (y = [ 0.0, 1.0, 2.0 ], )
```

!!! note
    The conditioning on data is a feature of `RxInfer`, not `GraphPPL`.

```@docs
RxInfer.condition_on
RxInfer.ConditionedModelGenerator
```

Sometimes it might be useful to condition on data, without actual data attached. 
This situation may occur during [reactive inference](@ref user-guide-infer-reactive-inference), when data becomes available _after_ model creation.

```@docs 
RxInfer.DefferedDataHandler
```

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

The `@model` macro automatically resolves any inner function calls into anonymous extra nodes. For example the following:
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

Note that the inference backend will try to optimize inner deterministic function calls in the case where all arguments are known in advance. 
For example:
```julia
y ~ Normal(mean = 0.0, variance = inv(2.0))
```
should not create an extra factor node for the `inv`, since `inv` is a deterministic function and all arguments are known in advance.

It is possible to use any functional expression within the `~` operator arguments list. 
The only one exception is the `ref` expression (e.g `x[i]`). 
Technically, in Julia, the `x[i]` call is translated to a function call `getindex(x, i)`. Thus the `@model` macro should create a factor node for the `getindex` function, but this won't happen in practice because this case is treated separately. This means that the model parser will not create unnecessary nodes when only simple indexing is involved. That also means that all expressions inside `x[...]` list are left untouched during model parsing. 

!!! note
    It is not allowed to use latent variables within square brackets in the model specification or for control flow statements such as `if`, `for` or `while`.

### Broadcasting syntax 

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

Note, however, that all variables that take part in the broadcasting operation must be defined in advance. That means that it is not possible to 
use broadcasting with [deffered data](@ref user-guide-model-specification-conditioning). Read more about how broadcasting machinery works in Julia in [the official documentation](https://docs.julialang.org/en/v1/manual/arrays/#Broadcasting).

### `Distributions.jl` compatibility

For some factor nodes we rely on the syntax from `Distributions.jl` to make it easy to adopt `RxInfer.jl` for these users. These nodes include for example the [`Beta`](https://en.wikipedia.org/wiki/Beta_distribution) and [`Wishart`](https://en.wikipedia.org/wiki/Wishart_distribution) distributions. These nodes can be created using the `~` syntax with the arguments as specified in `Distributions.jl`. Unfortunately, we `RxInfer.jl` is not yet compatible with all possible distributions to be used as factor nodes. If you feel that you would like to see another node implemented, please file an issue.

!!! note
    To quickly check the list of all available factor nodes that can be used in the model specification language call `?ReactiveMP.is_predefined_node` or `Base.doc(ReactiveMP.is_predefined_node)`.


Specifically for the Gaussian/Normal case we have custom implementations that yield a higher computational efficiency and improved stability in comparison to `Distributions.jl` as these are optimized for sampling operations. Our aliases for these distributions therefore do not correspond to the implementations from `Distributions.jl`. However, our model specification language is compatible with syntax from `Distributions.jl` for normal distributions, which will be automatically converted. `RxInfer` has its own implementation because of the following 3 reasons:
1. `Distributions.jl` constructs normal distributions by saving the corresponding covariance matrices in a `PDMat` object from `PDMats.jl`. This construction always computes the Cholesky decompositions of the covariance matrices, which is very convenient for sampling-based procedures. However, in `RxInfer.jl` we mostly base our computations on analytical expressions which do not always need to compute the Cholesky decomposition. In order to reduce the overhead that `Distributions.jl` introduces, we therefore have custom implementations.
2. Depending on the update rules, we might favor different parameterizations of the normal distributions. `ReactiveMP.jl` has quite a variety in parameterizations that allow us to efficient computations where we convert between parameterizations as little as possible.
3. In certain situations we value stability a lot, especially when inverting matrices. `PDMats.jl`, and hence `Distributions.jl`, is not capable to fulfill all needs that we have here. Therefore we use `PositiveFactorizations.jl` to cope with the corner-cases.

### Node creation options

To pass optional arguments to the node creation constructor the user can use the `where { options...  }` options specification syntax.

Example:

```julia
y ~ Normal(mean = y_mean, var = y_var) where { ... }
```

A list of the available options specific to the `ReactiveMP` inference engine is presented below.

#### Metadata option

Is is possible to pass any extra metadata to a factor node with the `meta` option. Metadata can be later accessed in message computation rules. See also [Meta specification](@ref user-guide-meta-specification) section.

```julia
z ~ f(x, y) where { meta = ... }
```

#### Depedencies option

A user can modify default computational pipeline of a node with the `dependencies` options. 
Read more about different options in the [`ReactiveMP.jl` documentation](https://reactivebayes.github.io/ReactiveMP.jl/stable/).

```julia
y[k - 1] ~ Probit(x[k]) where { dependencies = ... }
```

