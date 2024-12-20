# [Using `=` instead of `:=` for deterministic nodes](@id usage-colon-equality)

When specifying probabilistic models in RxInfer, you might be tempted to use the `=` operator for deterministic relationships between variables. While this may seem natural from a programming perspective (especially if you're coming from other frameworks - see [Comparison to other packages](@ref comparison)), it doesn't align with how Bayesian inference and factor graphs work. Let's explore why RxInfer uses a different approach and how it enables powerful probabilistic modeling.

## The Problem

Consider this seemingly reasonable model specification:

```julia
@model function wrong_model(θ)
    x ~ MvNormal(mean = [ 0.0, 0.0 ], cov = [ 1.0 0.0; 0.0 1.0 ])
    y = dot(x, θ)      # This won't work!
    z ~ Normal(y, 1.0)
end
```

This code will fail because:
1. During model creation, `x` is not an actual vector of numbers - it's a reference to a node in the factor graph
2. Julia's `dot` function expects a vector input, not a graph node
3. The `=` operator performs immediate assignment and executes the `dot` function, which isn't what we want for building factor graphs

## The Solution

Use the `:=` operator for deterministic relationships:

```julia
@model function correct_model()
    x ~ MvNormal(mean = [ 0.0, 0.0 ], cov = [ 1.0 0.0; 0.0 1.0 ])
    y := dot(x, θ)     # This is correct!
    z ~ Normal(y, 1.0)
end
```

The `:=` operator:
- Creates a deterministic node in the factor graph
- Properly tracks dependencies between variables
- Allows RxInfer to handle the computation during inference

!!! tip
    If you're coming from other probabilistic programming frameworks like Turing.jl, remember that RxInfer uses `:=` for deterministic relationships. While this might seem unusual at first, it's a deliberate design choice that enables powerful message-passing inference algorithms.

## Why Not `=`?

RxInfer's design is based on factor graphs, which are probabilistic graphical models that represent the factorization of a joint probability distribution. In a factor graph:

- Variables are represented as nodes (vertices) in the graph
- Factor nodes connect variables and encode their relationships
- Edges represent the dependencies between variables and factors
- Both probabilistic (`~`) and deterministic (`:=`) relationships create specific types of factor nodes

When you specify a model, RxInfer constructs this graph structure where:
- Each `~` creates a factor node representing that probability distribution
- Each `:=` creates a deterministic factor node representing that transformation
- Variables are automatically connected to their relevant factors
- The graph captures the complete probabilistic model structure

This explicit graph-based design brings several key benefits:
- **Efficient Message Passing**: The graph structure enables localized belief propagation, where each node only needs to communicate with its immediate neighbors
- **Lazy Evaluation**: Factor nodes compute messages only when needed during inference, avoiding unnecessary calculations
- **Flexible Inference**: The same graph structure can support different message-passing schedules and inference algorithms
- **Modular Updates**: Changes in one part of the graph only affect the connected components

Using `=` would break this design because:
- It executes computations immediately during model specification, before the graph is built
- It prevents RxInfer from properly tracking the probabilistic dependencies
- It makes message passing impossible since there's no graph structure to pass messages through

## Implementation Details

When you write:
```julia
y := dot(x, θ)
```

RxInfer creates:
1. A deterministic factor node representing the `dot` function with `x` and `θ` as arguments (edges)
2. Creates a node for `y` if it has not been created yet
3. Proper edges connecting `x` and `θ` to this node and this node to `y`
4. Message passing rules for propagating beliefs through this transformation

This structured approach enables efficient inference and maintains the mathematical rigor of the probabilistic model.

For more details about model specification, see the [Model Specification](@ref user-guide-model-specification) guide, particularly the section on [Deterministic relationships](@ref user-guide-model-specification-node-creation-deterministic).