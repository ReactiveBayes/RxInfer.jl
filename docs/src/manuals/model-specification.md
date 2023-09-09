# [Model Specification](@id user-guide-model-specification)

The `RxInfer.jl` package exports the `@model` macro for model specification. This `@model` macro accepts the model specification itself in a form of regular Julia function. For example: 

```julia
@model function model_name(model_arguments...; model_keyword_arguments...)
    # model specification here
    return ...
end
```

Model options, `model_arguments` and `model_keyword_arguments` are optional and may be omitted:

```julia
@model function model_name()
    # model specification here
    return ...
end
```

The `@model` macro returns a regular Julia function (in this example `model_name()`) which can be executed as usual. It returns a so-called model generator object, e.g:

```julia
@model function my_model(model_arguments...)
    # model specification here
    # ...
    return x, y
end
```

```julia
generator = my_model(model_arguments...)
```

In order to create an instance of the model object we should use the `create_model` function:

```julia
model, (x, y) = create_model(generator)
```

It is not necessary to return anything from the model, in that case `RxInfer.jl` will automatically inject `return nothing` to the end of the model function.

## A full example before diving in

Before presenting the details of the model specification syntax, an example of a probabilistic model is given.
Here is an example of a simple state space model with latent random variables `x` and noisy observations `y`:

```julia
@model function state_space_model(n_observations, noise_variance)

    c = constvar(1.0)
    x = randomvar(n_observations)
    y = datavar(Float64, n_observations)

    x[1] ~ NormalMeanVariance(0.0, 100.0)

    for i in 2:n_observations
       x[i] ~ x[i - 1] + c
       y[i] ~ NormalMeanVariance(x[i], noise_var)
    end

    return x, y
end
```
    
## Graph variables creation

### [Constants](@id user-guide-model-specification-constant-variables)

Even though any runtime constant passed to a model as a model argument will be automatically converted to a fixed constant, sometimes it might be useful to create constants by hand (e.g. to avoid copying large matrices across the model and to avoid extensive memory allocations).

You can create a constant within a model specification macro with `constvar()` function. For example:

```julia
@model function model_name(...)
    ...
    c = constvar(1.0)

    for i in 2:n
        x[i] ~ x[i - 1] + c # Reuse the same reference to a constant 1.0
    end
    ...
end
```

!!! note 
    `constvar()` function is supposed to be used only within the `@model` macro.

Additionally you can specify an extra `::ConstVariable` type for some of the model arguments. In this case macro automatically converts them to a single constant using `constvar()` function. E.g.:

```julia
@model function model_name(nsamples::Int, c::ConstVariable)
    ...
    # no need to call for a constvar() here
    for i in 2:n
        x[i] ~ x[i - 1] + c # Reuse the same reference to a constant `c`
    end
    ...
end
```

!!! note
    `::ConstVariable` annotation does not play role in Julia's multiple dispatch. `RxInfer.jl` removes this annotation and replaces it with `::Any`.

### [Data variables](@id user-guide-model-specification-data-variables)

It is important to have a mechanism to pass data values to the model. You can create data inputs with `datavar()` function. As a first argument it accepts a type specification and optional dimensionality (as additional arguments or as a tuple). User can treat `datavar()`s in the model as both clamped values for priors and observations.

Examples: 

```julia
@model function model_name(...)
    ...
    y = datavar(Float64) # Creates a single data input with `y` as identificator
    y = datavar(Float64, n) # Returns a vector of  `y_i` data input objects with length `n`
    y = datavar(Float64, n, m) # Returns a matrix of `y_i_j` data input objects with size `(n, m)`
    y = datavar(Float64, (n, m)) # It is also possible to use a tuple for dimensionality
    ...
end
```

!!! note 
    `datavar()` function is supposed to be used only within the `@model` macro.

`datavar()` call within `@model` macro supports `where { options... }` block for extra options specification, e.g:

```julia
@model function model_name(...)
    ...
    y = datavar(Float64, n) where { allow_missing = true }
    ...
end
```

#### Data variables available options

- `allow_missing = true/false`: Specifies if it is possible to pass `missing` object as an observation. Note however that by default the `ReactiveMP` inference engine does not expose any message computation rules that involve `missing`s.

### [Random variables](@id user-guide-model-specification-random-variables)

There are several ways to create random variables. The first one is an explicit call to `randomvar()` function. By default it doesn't accept any argument, creates a single random variable in the model and returns it. It is also possible to pass dimensionality arguments to `randomvar()` function in the same way as for the `datavar()` function.

Examples: 

```julia
@model function model_name(...)
    ...
    x = randomvar() # Returns a single random variable which can be used later in the model
    x = randomvar(n) # Returns an vector of random variables with length `n`
    x = randomvar(n, m) # Returns a matrix of random variables with size `(n, m)`
    x = randomvar((n, m)) # It is also possible to use a tuple for dimensionality
    ...
end
```

!!! note 
    `randomvar()` function is supposed to be used only within the `@model` macro.

`randomvar()` call within `@model` macro supports `where { options... }` block for extra options specification, e.g:

```julia
@model function model_name(...)
    ...
    y = randomvar() where { prod_constraint = ProdGeneric() }
    ...
end
```

#### Random variables available options

- `prod_constraint`
- `prod_strategy`
- `marginal_form_constraint`
- `marginal_form_check_strategy`
- `messages_form_constraint`
- `messages_form_check_strategy`
- `pipeline`

The second way to create a random variable is to create a node with the `~` operator. If the random variable has not yet been created before this call, it will be created automatically during the creation of the node. Read more about the `~` operator below.

## Node creation

Factor nodes are used to define a relationship between random variables and/or constants and data inputs. A factor node defines a probability distribution over selected random variables. 

### `Distributions.jl` compatibility

For some factor nodes we rely on the syntax from `Distributions.jl` to make it easy to adopt `RxInfer.jl` for these users. These nodes include for example the Beta and Wishart distributions. These nodes can be created using the `~` syntax with the arguments as specified in `Distributions.jl`. Unfortunately, we `RxInfer.jl` is not yet compatible with all possible distributions to be used as factor nodes. If you feel that you would like to see another node implemented, please file an issue.

!!! note
    To quickly check the list of all available factor nodes that can be used in the model specification language call `?make_node` or `Base.doc(make_node)`.


Specifically for the Gaussian/Normal case we have custom implementations that yield a higher computational efficiency and improved stability in comparison to `Distributions.jl` as these are optimized for sampling operations. Our aliases for these distributions therefore do not correspond to the implementations from `Distributions.jl`. However, our model specification language is compatible with syntax from `Distributions.jl` for normal distributions, which will be automatically converted. `RxInfer` has its own implementation because of the following 3 reasons:
1. `Distributions.jl` constructs normal distributions by saving the corresponding covariance matrices in a `PDMat` object from `PDMats.jl`. This construction always computes the Cholesky decompositions of the covariance matrices, which is very convenient for sampling-based procedures. However, in `RxInfer.jl` we mostly base our computations on analytical expressions which do not always need to compute the Cholesky decomposition. In order to reduce the overhead that `Distributions.jl` introduces, we therefore have custom implementations.
2. Depending on the update rules, we might favor different parameterizations of the normal distributions. `ReactiveMP.jl` has quite a variety in parameterizations that allow us to efficient computations where we convert between parameterizations as little as possible.
3. In certain situations we value stability a lot, especially when inverting matrices. `PDMats.jl`, and hence `Distributions.jl`, is not capable to fulfill all needs that we have here. Therefore we use `PositiveFactorizations.jl` to cope with the corner-cases.

### Tilde syntax

We model a random variable by a probability distribution using the `~` operator. For example, to create a random variable `y` which is modeled by a Normal distribution, where its mean and variance are controlled by the random variables `m` and `v` respectively, we define

```julia
@model function model_name(...)
    ...
    m = randomvar()
    v = randomvar()
    y ~ NormalMeanVariance(m, v) # Creates a `y` random variable automatically
    ...
end
```

Another example, but using a deterministic relation between random variables:

```julia
@model function model_name(...)
    ...
    a = randomvar()
    b = randomvar()
    c ~ a + b
    ...
end
```

!!! note
    The `RxInfer.jl` package uses the `~` operator for modelling both stochastic and deterministic relationships between random variables.


The `@model` macro automatically resolves any inner function calls into anonymous extra nodes in case this inner function call is a non-linear transformation. It will also create needed anonymous random variables. But it is important to note that the inference backend will try to optimize inner non-linear deterministic function calls in the case where all arguments are constants or data inputs. For example:

```julia
noise ~ NormalMeanVariance(mean, inv(precision)) # Will create a non-linear `inv` node in case if `precision` is a random variable. Won't create an additional non-linear node in case if `precision` is a constant or data input.
```

It is possible to use any functional expression within the `~` operator arguments list. The only one exception is the `ref` expression (e.g `x[i]`). All reference expressions within the `~` operator arguments list are left untouched during model parsing. This means that the model parser will not create unnecessary nodes when only simple indexing is involved.

!!! note
    It is forbidden to use random variable within square brackets in the model specification.

```julia
y ~ NormalMeanVariance(x[i - 1], variance) # While in principle `i - 1` is an inner function call (`-(i, 1)`) model parser will leave it untouched and won't create any anonymous nodes for `ref` expressions.

y ~ NormalMeanVariance(A * x[i - 1], variance) # This example will create a `*` anonymous node (in case if x[i - 1] is a random variable) and leave `x[i - 1]` untouched.
```

It is also possible to return a node reference from the `~` operator. Use the following syntax:

```julia
node, y ~ NormalMeanVariance(mean, var)
```

Having a node reference can be useful in case the user wants to return it from a model and to use it later on to specify initial joint marginal distributions.

### Broadcasting syntax 

!!! note 
    Broadcasting syntax requires at least v2.1.0 of `GraphPPL.jl` 

GraphPPL support broadcasting for `~` operator in the exact same way as Julia itself. A user is free to write an expression of the following form:

```julia
y = datavar(Float64, n)
y .~ NormalMeanVariance(0.0, 1.0) # <- i.i.d observations
```

More complex expression are also allowed:

```julia
m ~ NormalMeanPrecision(0.0, 0.0001)
t ~ Gamma(1.0, 1.0)

y = randomvar(Float64, n)
y .~ NormalMeanPrecision(m, t)
```

```julia
A = constvar(...)
x = randomvar(n)
y = datavar(Vector{Float64}, n)

w         ~ Wishart(3, diageye(2))
x[1]      ~ MvNormalMeanPrecision(zeros(2), diageye(2))
x[2:end] .~ A .* x[1:end-1] # <- State-space model with transition matrix A
y        .~ MvNormalMeanPrecision(x, w) # <- Observations with unknown precision matrix
```

Note, however, that all variables that take part in the broadcasting operation must be defined before either with `randomvar` or `datavar`. The exception here is constants that are automatically converted to their `constvar` equivalent. If you want to prevent broadcasting for some constant (e.g. if you want to add a vector to a multivariate Gaussian distribution) use explicit `constvar` call:

```julia
# Suppose `x` is a 2-dimensional Gaussian distribution
z .~ x .+ constvar([ 1, 1 ])
# Which is equivalent to 
for i in 1:n
   z[i] ~ x[i] + constvar([ 1, 1 ])
end
```

Without explicit `constvar` Julia's broadcasting machinery would instead attempt to unroll for loop in the following way:

```julia
# Without explicit `constvar`
z .~ x .+ [ 1, 1 ]
# Which is equivalent to 
array = [1, 1]
for i in 1:n
   z[i] ~ x[i] + array[i] # This is wrong if `x[i]` is supposed to be a multivariate Gaussian 
end
```

Read more about how broadcasting machinery works in Julia in [the official documentation](https://docs.julialang.org/en/v1/manual/arrays/#Broadcasting).

### Node creation options

To pass optional arguments to the node creation constructor the user can use the `where { options...  }` options specification syntax.

Example:

```julia
y ~ NormalMeanVariance(y_mean, y_var) where { q = q(y_mean)q(y_var)q(y) } # mean-field factorisation over q
```

A list of the available options specific to the `ReactiveMP` inference engine is presented below.

#### Factorisation constraint option

See also [Constraints Specification](@ref user-guide-constraints-specification) section.

Users can specify a factorisation constraint over the approximate posterior `q` for variational inference.
The general syntax for factorisation constraints over `q` is the following:
```julia
variable ~ Node(node_arguments...) where { q = RecognitionFactorisationConstraint }
```

where `RecognitionFactorisationConstraint` can be the following

1. `MeanField()`

Automatically specifies a mean-field factorisation

Example:

```julia
y ~ NormalMeanVariance(y_mean, y_var) where { q = MeanField() }
```

2. `FullFactorisation()`

Automatically specifies a full factorisation

Example:

```julia
y ~ NormalMeanVariance(y_mean, y_var) where { q = FullFactorisation() }
```

3. `q(μ)q(v)q(out)` or `q(μ) * q(v) * q(out)`

A user can specify any factorisation he wants as the multiplication of `q(interface_names...)` factors. As interface names the user can use the interface names of an actual node (read node's documentation), its aliases (if available) or actual random variable names present in the `~` operator expression.

Examples: 

```julia
# Using interface names of a `NormalMeanVariance` node for factorisation constraint. 
# Call `?NormalMeanVariance` to know more about interface names for some node
y ~ NormalMeanVariance(y_mean, y_var) where { q = q(μ)q(v)q(out) }
y ~ NormalMeanVariance(y_mean, y_var) where { q = q(μ, v)q(out) }

# Using interface names aliases of a `NormalMeanVariance` node for factorisation constraint. 
# Call `?NormalMeanVariance` to know more about interface names aliases for some node
# In general aliases correspond to the function names for distribution parameters
y ~ NormalMeanVariance(y_mean, y_var) where { q = q(mean)q(var)q(out) }
y ~ NormalMeanVariance(y_mean, y_var) where { q = q(mean, var)q(out) }

# Using random variables names from `~` operator expression
y ~ NormalMeanVariance(y_mean, y_var) where { q = q(y_mean)q(y_var)q(y) }
y ~ NormalMeanVariance(y_mean, y_var) where { q = q(y_mean, y_var)q(y) }

# All methods can be combined easily
y ~ NormalMeanVariance(y_mean, y_var) where { q = q(μ)q(y_var)q(out) }
y ~ NormalMeanVariance(y_mean, y_var) where { q = q(y_mean, v)q(y) }
```

#### Metadata option

Is is possible to pass any extra metadata to a factor node with the `meta` option. Metadata can be later accessed in message computation rules. See also [Meta specification](@ref user-guide-meta-specification) section.

```julia
z ~ f(x, y) where { meta = ... }
```
