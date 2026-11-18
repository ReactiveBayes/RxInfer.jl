# [Deterministic nodes](@id delta-node-manual)

Most nodes of RxInfer.jl are distributions, mainly from the exponential family, and compositions
of them, such as the Gaussian controlled variance (GCV) and autoregressive (AR) nodes. A
deterministic transformation of one or several random variables, `y := f(x)`, is a node too:
the *delta node*. Its messages have no closed form for an arbitrary `f`, so the node needs an
approximation method, which you choose per node. This guide describes the methods and when each
applies.

## Features and supported inference scenarios

The delta node supports two approximation methods, and a node without a method can run under a
fallback scheme. Which option fits depends on the nodes around the delta node:

1. **Gaussian nodes**: for a delta node connected only to univariate or multivariate Gaussian
   distributions, use [`Linearization`](@extref MessagePassingRulesApproximations.Linearization)
   or [`Unscented`](@extref MessagePassingRulesApproximations.Unscented).
2. **Exponential family nodes**: for a delta node connected to other members of the exponential
   family, give it no method and run the inference with `fallback = NaturalGradientPropagation()`.
   This scheme approximates each variable around the node within its own family, such as a Gamma
   or a Beta (see [Exponential family case](@ref delta-node-exponential-family)).
3. **Stacked delta nodes**: for delta nodes connected to each other, any of them applies.
4. **Inverse functions**: when the inverse of `f` is known, `Linearization` and `Unscented` use it.

| Method        | Gaussian nodes | Exponential family nodes | Stacked delta nodes | Inverse functions |
|---------------|----------------|--------------------------|---------------------|-------------------|
| Linearization | ✓              | ✗                        | ✓                   | ✓                 |
| Unscented     | ✓              | ✗                        | ✓                   | ✓                 |
| no method, `fallback = NaturalGradientPropagation()` | ✓ | ✓              | ✓                   | ✗                 |

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

## [Exponential family case](@id delta-node-exponential-family)

When the delta node is connected to nodes of the exponential family other than Gaussians,
`Linearization` and `Unscented` do not apply. Give the node no method, and run the inference with
the fallback scheme [`NaturalGradientPropagation`](@ref). Every variable around the node then keeps
one site per edge in its own family, the family of the rest of its belief. The sites move by damped
natural-gradient steps towards the variational optimum over that family. Here is a modified
example, with a Gamma prior on `x` and Bernoulli observations of `tanh(x)`:

```@example delta_node_example_ngmp
using RxInfer

@model function delta_node_example1(z)
    x ~ Gamma(shape = 1.0, rate = 1.0)
    y := tanh(x)
    z .~ Bernoulli(y)
end

observations = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
result = infer(
    model      = delta_node_example1(),
    data       = (z = observations,),
    iterations = 50,
    fallback   = NaturalGradientPropagation(),
)

(x = last(result.posteriors[:x]), y = last(result.posteriors[:y]))
```

The Bernoulli observations send Beta messages towards `y`, so the sites of `y` are Beta
distributions. The message of the node towards `y`, which is the prior of `x` pushed forward through
`tanh`, gets a Beta site. The message of the node towards `x`, which is the Beta message pulled back
through `tanh`, gets a Gamma site, because the prior of `x` is a Gamma. The posteriors are a Gamma
and a Beta distribution, and their means are close to those of the exact posterior computed on a
grid:

```@example delta_node_example_ngmp
grid = range(1e-6, 30; length = 100_000)
logposterior = [logpdf(Gamma(1.0, 1.0), x) + sum(z == 1 ? log(tanh(x)) : log1p(-tanh(x)) for z in observations) for x in grid]
weights = exp.(logposterior .- maximum(logposterior))
weights ./= sum(weights)
exact = (x = sum(weights .* grid), y = sum(weights .* tanh.(grid)))
using Test #hide
@test last(result.posteriors[:x]) isa GammaShapeRate && last(result.posteriors[:y]) isa Beta #hide
@test isapprox(mean(last(result.posteriors[:x])), exact.x, rtol = 1e-2) #hide
@test isapprox(mean(last(result.posteriors[:y])), exact.y, rtol = 1e-2) #hide
(mean(last(result.posteriors[:x])), exact.x), (mean(last(result.posteriors[:y])), exact.y)
```

The scheme does not sample, and it needs no constraints or initialization, because the sites start
from the priors. [Natural-gradient propagation](@ref inference-undefinedrules-ngmp) describes the
scheme. The `CVIProjection` method of DeltaMessagePassingRules, which projected the node's messages
by sampling, is deprecated in favor of this scheme.

## Fuse deterministic nodes with stochastic nodes

You can also avoid the approximation altogether, by fusing the deterministic relation with a
neighboring stochastic node, as [this section](@ref inference-undefinedrules-fusedelta) shows.
