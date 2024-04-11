# [Deterministic nodes](@id delta-node-manual)

RxInfer.jl offers a comprehensive set of stochastic nodes, with a primary emphasis on distributions from the exponential family and its associated compositions, such as Gaussian with controlled variance (GCV) or autoregressive (AR) nodes. The `DeltaNode` stands out in this package, representing a deterministic transformation of either a single random variable or a group of them. This guide provides insights into the `DeltaNode` and its functionalities.

## Features and Supported Inference Scenarios

The delta node has several approximation methods for performing probabilistic inference. The desired approximation method depends on the nodes connected to the delta node. We differentiate the following deterministic transformation scenarios:

1. **Gaussian Nodes**: For delta nodes linked to strictly multivariate or univariate Gaussian distributions, the recommended methods are Linearization or Unscented transforms.
2. **Exponential Family Nodes**: For the delta node connected to nodes from the exponential family, the CVI (Conjugate Variational Inference) is the method of choice.
3. **Stacking Delta Nodes**: For scenarios where delta nodes are stacked, either Linearization or Unscented transforms are suitable.

The table below summarizes the features of the delta node in RxInfer.jl, categorized by the approximation method:

| Methods       | Gaussian Nodes | Exponential Family Nodes | Stacking Delta Nodes 
|---------------|----------------|--------------------------|----------------------
| Linearization | ✓              | ✗                        | ✓                    
| Unscented     | ✓              | ✗                        | ✓                    
| CVI           | ✓              | ✓                        | ✗                    

## Gaussian Case

In the context of Gaussian distributions, we recommend either the `Linearization` or `Unscented` method for delta node approximation. The `Linearization` method provides a first-order approximation, while the `Unscented` method delivers a more precise second-order approximation. It's worth noting that while the `Unscented` method is more accurate, it may require hyper-parameters tuning.


For clarity, consider the following example:

```@example delta_node_example
using RxInfer

@model function delta_node_example(z)
    x ~ Normal(mean=0.0, var=1.0)
    y := tanh(x)
    z ~ Normal(mean=y, var=1.0)
end
```

To perform inference on this model, designate the approximation method for the delta node (here, the `tanh` function) using the `@meta` specification:

```@example delta_node_example
delta_meta = @meta begin 
    tanh() -> Linearization()
end
```
or
```@example delta_node_example
delta_meta = @meta begin 
    tanh() -> Unscented()
end
```

For a deeper understanding of the `Unscented` method and its parameters, consult the docstrings.

Given the invertibility of `tanh`, indicating its inverse function can optimize the inference procedure:

```@example delta_node_example
delta_meta = @meta begin 
    tanh() -> DeltaMeta(method = Linearization(), inverse = atanh)
end
```

To execute the inference procedure:

```@example delta_node_example
infer(model = delta_node_example(), meta=delta_meta, data = (z = 1.0,))
```

This methodology is consistent even when the delta node is associated with multiple nodes. For instance:

```@example delta_node_example
f(x, g) = x*tanh(g)
```

```@example delta_node_example
@model function delta_node_example(z)
    x ~ Normal(mean=1.0, var=1.0)
    g ~ Normal(mean=1.0, var=1.0)
    y := f(x, g)
    z ~ Normal(mean=y, var=0.1)
end
```

The corresponding meta specification is:

```@example delta_node_example
delta_meta = @meta begin 
    f() -> DeltaMeta(method = Linearization())
end
```
or simply
```@example delta_node_example
delta_meta = @meta begin 
    f() -> Linearization()
end
```

If specific functions outline the backward relation of variables within the `f` function, you can provide a tuple of inverse functions in the order of the variables:

```@example delta_node_example
f_back_x(out, g) = out/tanh(g)
f_back_g(out, x) = atanh(out/x)
```


```@example delta_node_example
delta_meta = @meta begin 
    f() -> DeltaMeta(method = Linearization(), inverse=(f_back_x, f_back_g))
end
```

## Exponential Family Case

When the delta node is associated with nodes from the exponential family (excluding Gaussians), the `Linearization` and `Unscented` methods are not applicable. In such cases, the CVI (Conjugate Variational Inference) is available. Here's a modified example:

```@example delta_node_example_cvi
using RxInfer

@model function delta_node_example1(z)
    x ~ Gamma(shape=1.0, rate=1.0)
    y := tanh(x)
    z ~ Bernoulli(y)
end
```

The corresponding meta specification can be represented as:

```@example delta_node_example_cvi
using StableRNGs
using Optimisers

delta_meta = @meta begin 
    tanh() -> DeltaMeta(method = CVI(StableRNG(42), 100, 100, Optimisers.Descent(0.01)))
end
```

Consult the `ProdCVI` docstrings for a detailed explanation of these parameters.