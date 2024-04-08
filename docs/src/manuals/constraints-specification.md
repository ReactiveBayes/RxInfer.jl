# [Constraints Specification](@id user-guide-constraints-specification)

`RxInfer.jl` uses a macro called `@constraints` from `GraphPPL` to add extra constraints during the inference process. For details on how to use the `@constraints` macro, you can check out the [official documentation](https://reactivebayes.github.io/GraphPPL.jl/stable/) of GraphPPL.

## [Background and example](@id user-guide-constraints-specification-background)

Here we briefly cover the mathematical aspects of constraints specification. For additional information and relevant links, please refer to the [Bethe Free Energy](@ref lib-bethe-free-energy) section. In essence, `RxInfer` performs Variational Inference (via message passing) given specific constraints $\mathcal{Q}$:

$$q^* = \arg\min_{q(s) \in \mathcal{Q}}F[q](\hat{y}) = \mathbb{E}_{q(s)}\left[\log \frac{q(s)}{p(s, y=\hat{y})} \right]\,.$$

The [`@model`](@ref) macro specifies generative model `p(s, y)` where `s` is a set of random variables and `y` is a set of observations. In a nutshell the goal of probabilistic programming is to find `p(s|y)`. `RxInfer` approximates `p(s|y)` with a proxy distribution `q(x)` using KL divergence and Bethe Free Energy optimisation procedure. By default there are no extra factorisation constraints on `q(s)` and the optimal solution is `q(s) = p(s|y)`.

For certain problems, it may be necessary to adjust the set of constraints $\mathcal{Q}$ (also known as the variational family of distributions) to either improve accuracy at the expense of computational resources or reduce accuracy to conserve computational resources. Sometimes, we are compelled to impose certain constraints because otherwise, the problem becomes too challenging to solve within a reasonable timeframe.

For instance, consider the following model:

```@example constraints-specification
using RxInfer

@model function iid_normal(y)
    μ  ~ Normal(mean = 0.0, variance = 1.0)
    τ  ~ Gamma(shape = 1.0, rate = 1.0)
    y .~ Normal(mean = μ, precision = τ)
end
```

In this model, we characterize all observations in a dataset y as a `Normal` distribution with mean `μ` and precision `τ`. It's reasonable to assume that the latent variables `μ` and `τ` are jointly independent, thereby rendering their joint posterior distribution as:

$$q(μ, τ) = q(μ)q(τ)\,.$$

If we would write the variational family of distribution for such an assumption, it would be expressed as:

$$\mathcal{Q} = \left\{ q : q(μ, τ) = q(μ)q(τ) \right\}\,.$$

We can express this constraint with the `@constraints` macro:

```@example constraints-specification
constraints = @constraints begin 
    q(μ, τ) = q(μ)q(τ)
end
```

and use the created `constraints` object to the [`infer`](@ref) function:

```@example constraints-specification
result = infer(
    model       = iid_normal(),
    # Sample data from mean `3.1415` and precision `2.7182`
    data        = (y = rand(NormalMeanPrecision(3.1415, 2.7182), 1000), ),
    constraints = constraints,
    # We need to specify initial marginals, since with the constraints 
    # the problem becomes inherently iterative (we could also specify initial for the mean instead)
    initmarginals = (τ = vague(Gamma), ),
    iterations    = 25
)
```

```@example constraints-specification
println("Estimated mean of `μ` is ", mean(result.posteriors[:μ][end]), " with standard deviation ", std(result.posteriors[:μ][end]))
println("Estimated mean of `τ` is ", mean(result.posteriors[:τ][end]), " with standard deviation ", std(result.posteriors[:τ][end]))
```

We observe that the estimates tend to slightly deviate from what the real values are. 
This behavior is a known characteristic of inference with the aforementioned constraints, often referred to as _Mean Field_ constraints.

## General syntax 

You can use the `@constraints` macro with either a regular Julia function or a single `begin ... end` block. Both ways are valid, as shown below:

```@example manual_constraints
using RxInfer #hide

# `functional` style
@constraints function create_my_constraints()
    q(μ, τ) = q(μ)q(τ)
end

# `block` style
myconstraints = @constraints begin 
    q(μ, τ) = q(μ)q(τ)
end

nothing #hide
```

The function-based syntax can also take arguments, like this:

```@example manual_constraints
@constraints function make_constraints(mean_field)
    # Specify mean-field only if the flag is `true`
    if mean_field
        q(μ, τ) = q(μ)q(τ)
    end
end

myconstraints = make_constraints(true)
```

## Marginal and messages form constraints

To specify marginal or messages form constraints `@constraints` macro uses `::` operator (in somewhat similar way as Julia uses it for multiple dispatch type specification).
Read more about available functional form constraints in the [Built-In Functional Forms](@ref lib-forms) section.

As an example, the following constraint:

```@example manual_constraints
@constraints begin 
    q(x) :: PointMassFormConstraint()
end
```

indicates that the resulting marginal of the variable (or array of variables) named `x` must be approximated with a `PointMass` object. Message passing based algorithms compute posterior marginals as a normalized product of two colliding messages on corresponding edges of a factor graph. In a few words `q(x)::PointMassFormConstraint` reads as:

```math
\mathrm{approximate~} q(x) = \frac{\overrightarrow{\mu}(x)\overleftarrow{\mu}(x)}{\int \overrightarrow{\mu}(x)\overleftarrow{\mu}(x) \mathrm{d}x}\mathrm{~as~PointMass}
```

Sometimes it might be useful to set a functional form constraint on messages too. For example if it is essential to keep a specific Gaussian parametrisation or if some messages are intractable and need approximation. To set messages form constraint `@constraints` macro uses `μ(...)` instead of `q(...)`:

```@example manual_constraints
@constraints begin 
    q(x) :: PointMassFormConstraint()
    μ(x) :: SampleListFormConstraint(1000)
    # it is possible to assign different form constraints on the same variable 
    # both for the marginal and for the messages 
end
```

`@constraints` macro understands "stacked" form constraints. For example the following form constraint

```@example manual_constraints
@constraints begin 
    q(x) :: (SampleListFormConstraint(1000) + PointMassFormConstraint())
end
```

indicates that the `q(x)` first must be approximated with a `SampleList` and in addition the result of this approximation should be approximated as a `PointMass`. 

!!! note
    Not all combinations of "stacked" form constraints are compatible between each other.

You can find more information about built-in functional form constraint in the [Built-in Functional Forms](@ref lib-forms) section. In addition, the [ReactiveMP library documentation](https://reactivebayes.github.io/ReactiveMP.jl/stable/) explains the functional form interfaces and shows how to build a custom functional form constraint that is compatible with `RxInfer.jl` and `ReactiveMP.jl` inference engine.

## Factorisation constraints on posterior distribution `q`

As has been mentioned [above](@ref user-guide-constraints-specification-background), inference may be not tractable for every model without extra factorisation constraints. To circumvent this, `RxInfer.jl` allows for extra factorisation constraints, for example:

```@example manual_constraints
@constraints begin 
    q(x, y) = q(x)q(y)
end
```

specifies a so-called mean-field assumption on variables `x` and `y` in the model. Furthermore, if `x` is an array of variables in our model we may induce extra mean-field assumption on `x` in the following way.

```@example manual_constraints
@constraints begin 
    q(x) = q(x[begin])..q(x[end])
    q(x, y) = q(x)q(y)
end
```

These constraints specify a mean-field assumption between variables `x` and `y` (either single variable or collection of variables) and additionally specify mean-field assumption on variables $x_i$.

!!! note 
    `@constraints` macro does not support matrix-based collections of variables. E.g. it is not possible to write `q(x[begin, begin])..q(x[end, end])`. Use `q(x[begin])..q(x[end])` instead.

Read more about the `@constraints` macro in the [official documentation](https://reactivebayes.github.io/GraphPPL.jl/stable/) of GraphPPL