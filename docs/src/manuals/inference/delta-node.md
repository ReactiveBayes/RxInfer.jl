# [Deterministic nodes](@id delta-node-manual)

Most nodes of RxInfer.jl are distributions, mainly from the exponential family, and compositions
of them, such as the Gaussian controlled variance (GCV) and autoregressive (AR) nodes. A
deterministic transformation of one or several random variables, `y := f(x)`, is a node too:
the *delta node*. Its messages have no closed form for an arbitrary `f`, so the node needs an
approximation method, which you choose per node. This guide describes the methods and when each
applies.

## Features and supported inference scenarios

The delta node supports three approximation methods. Which one fits depends on the nodes around
the delta node:

1. **Gaussian nodes**: for a delta node connected only to univariate or multivariate Gaussian
   distributions, use [`Linearization`](@extref MessagePassingRulesApproximations.Linearization)
   or [`Unscented`](@extref MessagePassingRulesApproximations.Unscented).
2. **Exponential family nodes**: for a delta node connected to other members of the exponential
   family, use [`CVIProjection`](@extref DeltaMessagePassingRules.CVIProjection).
3. **Stacked delta nodes**: for delta nodes connected to each other, any of the three methods
   applies.
4. **Inverse functions**: when the inverse of `f` is known, `Linearization` and `Unscented` use it.

| Method        | Gaussian nodes | Exponential family nodes | Stacked delta nodes | Inverse functions |
|---------------|----------------|--------------------------|---------------------|-------------------|
| Linearization | ✓              | ✗                        | ✓                   | ✓                 |
| Unscented     | ✓              | ✗                        | ✓                   | ✓                 |
| CVIProjection | ✓              | ✓                        | ✓                   | ✗                 |

The node's [algorithm](@extref MessagePassingRulesBase glossary-algorithm) is
[`DeltaApproximation`](@extref DeltaMessagePassingRules.DeltaApproximation), which carries the
method and, optionally, the inverse. You give it to the node with `@algorithm`, as
[Algorithm specification](@ref user-guide-algorithm-specification) describes.

## Gaussian case

For Gaussian distributions, use either `Linearization` or `Unscented`. `Linearization` is a
first-order approximation. `Unscented` is a more precise second-order approximation, and it may
need its hyperparameters tuned. Both methods work well for a differentiable function; for a
function that is not differentiable, their results may be inaccurate.

Consider the following example:

```@example delta_node_example
using RxInfer

@model function delta_node_example(z)
    x  ~ Normal(mean = 0.0, var = 1.0)
    y := tanh(x)
    z  ~ Normal(mean = y, var = 1.0)
end
```

!!! note
    It is advised, though not required, to write a deterministic relationship with `:=` in the
    `@model` macro.

To run inference in this model, give the delta node, here the `tanh` function, its approximation
method with `@algorithm`:

```@example delta_node_example
delta_algorithm = @algorithm begin
    tanh() -> DeltaApproximation(method = Linearization())
end
nothing # hide
```

A method alone is a shorthand for `DeltaApproximation(method = ...)`:

```@example delta_node_example
delta_algorithm = @algorithm begin
    tanh() -> Unscented()
end
nothing # hide
```

The docstrings of [`Unscented`](@extref MessagePassingRulesApproximations.Unscented) and
[`Linearization`](@extref MessagePassingRulesApproximations.Linearization) describe their
parameters.

`tanh` is invertible, and giving its inverse lets the node compute the message towards `x`
directly from the message from `z`:

```@example delta_node_example
delta_algorithm = @algorithm begin
    tanh() -> DeltaApproximation(method = Linearization(), inverse = atanh)
end
nothing # hide
```

Pass the specification to `infer` as `algorithm`:

```@example delta_node_example
result = infer(
    model     = delta_node_example(),
    algorithm = delta_algorithm,
    data      = (z = 1.0,),
)
```

The same holds for a delta node with several inputs. For instance:

```@example delta_node_example
f(x, g) = x * tanh(g)
```

```@example delta_node_example
@model function delta_node_example(z)
    x ~ Normal(mean = 1.0, var = 1.0)
    g ~ Normal(mean = 1.0, var = 1.0)
    y := f(x, g)
    z ~ Normal(mean = y, var = 0.1)
end
```

The corresponding algorithm specification is

```@example delta_node_example
delta_algorithm = @algorithm begin
    f() -> DeltaApproximation(method = Linearization())
end
nothing # hide
```

or, with the shorthand,

```@example delta_node_example
delta_algorithm = @algorithm begin
    f() -> Linearization()
end

result = infer(model = delta_node_example(), algorithm = delta_algorithm, data = (z = 1.0,))
```

When functions express each input of `f` in terms of the output and the other inputs, you can
give them as a tuple of inverses, in the order of the inputs:

```@example delta_node_example
f_back_x(out, g) = out / tanh(g)
f_back_g(out, x) = atanh(out / x)
```

```@example delta_node_example
delta_algorithm = @algorithm begin
    f() -> DeltaApproximation(method = Linearization(), inverse = (f_back_x, f_back_g))
end

result = infer(model = delta_node_example(), algorithm = delta_algorithm, data = (z = 1.0,))
```

## Exponential family case

When the delta node is connected to nodes of the exponential family other than Gaussians,
`Linearization` and `Unscented` do not apply. `CVIProjection` does: it projects the node's
messages onto members of the exponential family by stochastic optimization. Here is a modified
example:

!!! note
    The `CVIProjection` method is available only when the `ExponentialFamilyProjection` package
    is loaded in the current environment.

```@example delta_node_example_cvi
using RxInfer, ExponentialFamilyProjection, StableRNGs

@model function delta_node_example1(z)
    x ~ Gamma(shape = 1.0, rate = 1.0)
    y := tanh(x)
    z .~ Bernoulli(y)
end
```

`CVIProjection` projects onto the families that you name with `ProjectedTo` in `@constraints`,
and its rules read the marginal of the output, which needs an initial value:

```@example delta_node_example_cvi
delta_algorithm = @algorithm begin
    tanh() -> DeltaApproximation(method = CVIProjection())
end

delta_constraints = @constraints begin
    q(x)::ProjectedTo(Gamma)
    q(y)::ProjectedTo(Beta)
end

delta_initialization = @initialization begin
    q(y) = Beta(1.0, 1.0)
end
nothing # hide
```

`CVIProjection` samples. It draws from the random number generator of the engine, which you
choose with the `context` option of `infer`, here for reproducible results:

```@example delta_node_example_cvi
result = infer(
    model          = delta_node_example1(),
    algorithm      = delta_algorithm,
    constraints    = delta_constraints,
    initialization = delta_initialization,
    data           = (z = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],),
    iterations     = 10,
    options        = (context = (rng = StableRNG(42),),),
)

(x = result.posteriors[:x][end], y = result.posteriors[:y][end])
```

The docstring of [`CVIProjection`](@extref DeltaMessagePassingRules.CVIProjection) explains its
hyperparameters. Also read the [Non-conjugate Inference](@ref inference-nonconjugate) section.

## Fuse deterministic nodes with stochastic nodes

You can also avoid the approximation altogether, by fusing the deterministic relation with a
neighboring stochastic node, as [this section](@ref inference-undefinedrules-fusedelta) shows.
