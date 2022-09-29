# [Constraints Specification](@id user-guide-constraints-specification)

`RxInfer.jl` exports `@constraints` macro for the extra constraints specification that can be used during the inference step in `ReactiveMP.jl` engine package.

## General syntax 

`@constraints` macro accepts either regular Julia function or a single `begin ... end` block. For example both are valid:

```julia

# `functional` style
@constraints function create_my_constraints(arg1, arg2)
    ...
end

# `block` style
myconstraints = @constraints begin 
    ...
end

```

In the first case it returns a function that return constraints upon calling, e.g. 

```@example manual_constraints
@constraints function make_constraints(mean_field)
    q(x) :: PointMass

    if mean_field
        q(x, y) = q(x)q(y)
    end
end

myconstraints = make_constraints(true)
```
 
and in the second case it evaluates automatically and returns constraints object directly.

```@example manual_constraints
myconstraints = @constraints begin 
    q(x) :: PointMass
    q(x, y) = q(x)q(y)
end
```

### Options specification 

`@constraints` macro accepts optional list of options as a first argument and specified as an array of `key = value` pairs, e.g. 

```julia
myconstraints = @constraints [ warn = false ] begin 
   ...
end
```

List of available options:
- `warn::Bool` - enables/disables various warnings with an incompatible model/constraints specification

## Marginal and messages form constraints

To specify marginal or messages form constraints `@constraints` macro uses `::` operator (in somewhat similar way as Julia uses it for multiple dispatch type specification)

The following constraint:

```@example manual_constraints
@constraints begin 
    q(x) :: PointMass
end
```

indicates that the resulting marginal of the variable (or array of variables) named `x` must be approximated with a `PointMass` object. Message passing based algorithms compute posterior marginals as a normalized product of two colliding messages on corresponding edges of a factor graph. In a few words `q(x)::PointMass` reads as:

```math
\mathrm{approximate~} q(x) = \frac{\overrightarrow{\mu}(x)\overleftarrow{\mu}(x)}{\int \overrightarrow{\mu}(x)\overleftarrow{\mu}(x) \mathrm{d}x}\mathrm{~as~PointMass}
```

Sometimes it might be useful to set a functional form constraint on messages too. For example if it is essential to keep a specific Gaussian parametrisation or if some messages are intractable and need approximation. To set messages form constraint `@constraints` macro uses `μ(...)` instead of `q(...)`:

```@example manual_constraints
@constraints begin 
    q(x) :: PointMass
    μ(x) :: SampleList 
    # it is possible to assign different form constraints on the same variable 
    # both for the marginal and for the messages 
end
```

`@constraints` macro understands "stacked" form constraints. For example the following form constraint

```@example manual_constraints
@constraints begin 
    q(x) :: SampleList(1000) :: PointMass
end
```

indicates that the `q(x)` first must be approximated with a `SampleList` and in addition the result of this approximation should be approximated as a `PointMass`. 

!!! note
    Not all combinations of "stacked" form constraints are compatible between each other.

You can find more information about built-in functional form constraint in the [Built-in Functional Forms](@ref lib-forms) section. In addition, the [ReactiveMP library documentation](https://biaslab.github.io/ReactiveMP.jl/stable/) explains the functional form interfaces and shows how to build a custom functional form constraint that is compatible with `RxInfer.jl` and `ReactiveMP.jl` inference engine.

## Factorisation constraints on posterior distribution `q()`

`@model` macro specifies generative model `p(s, y)` where `s` is a set of random variables and `y` is a set of observations. In a nutshell the goal of probabilistic programming is to find `p(s|y)`. `RxInfer` approximates `p(s|y)` with a proxy distribution `q(x)` using KL divergence and Bethe Free Energy optimisation procedure. By default there are no extra factorisation constraints on `q(s)` and the optimal solution is `q(s) = p(s|y)`. However, inference may be not tractable for every model without extra factorisation constraints. To circumvent this, `RxInfer.jl` and `ReactiveMP.jl` accept optional factorisation constraints specification syntax:

For example:

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
    `@constraints` macro does not support matrix-based collections of variables. E.g. it is not possible to write `q(x[begin, begin])..q(x[end, end])`

It is possible to write more complex factorisation constraints, for example:

```@example manual_constraints
@constraints begin 
    q(x, y) = q(x[begin], y[begin])..q(x[end], y[end])
end
```

specifies a mean-field assumption between collection of variables named `x` and `y` only for variables with different indices. Another example is

```@example manual_constraints
@constraints function make_constraints(k)
    q(x) = q(x[begin:k])q(x[k+1:end])
end
```

In this example we specify a mean-field assumption between a set of variables `x[begin:k]` and `x[k+1:end]`. 

To create a model with extra constraints the user may pass an optional `constraints` positional argument for the model generator function:

```julia
@model function my_model(arguments...)
   ...
end

constraints = @constraints begin 
    ...
end

model, returnval = my_model(arguments...)(; constraints = constraints)
```

Alternatively, it is possible to use constraints directly in the automatic [`inference`](@ref) and [`rxinference`](@ref) functions that accepts `constraints` keyword argument. 