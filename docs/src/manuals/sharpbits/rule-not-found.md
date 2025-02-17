# [Rule Not Found Error](@id rule-not-found)

When using RxInfer, you might encounter a `RuleNotFoundError`. This error occurs during message-passing inference when the system cannot find appropriate update rules for computing messages between nodes in your factor graph. Let's understand why this happens and how to resolve it.

## Why does this happen?

Message-passing inference works by exchanging messages between nodes in a factor graph. Each message represents a probability distribution, and the rules for computing these messages depend on:

1. The type of the factor node (e.g., `Normal`, `Gamma`, etc.)
2. The types of incoming messages (e.g., `Normal`, `PointMass`, etc.) 
3. The interface through which the message is being computed
4. The inference method being used (Belief Propagation or Variational Message Passing)

The last point is particularly important - some message update rules may exist for Variational Message Passing (VMP) but not for Belief Propagation (BP), or vice versa. This is because BP aims to compute exact posterior distributions through message passing (when possible), while VMP approximates the posterior using the Bethe approximation. For a detailed mathematical treatment of these differences, see our [Bethe Free Energy implementation](@ref lib-bethe-free-energy) guide.

For example, consider this simple model:

```julia
@model function problematic_model()
    μ ~ Normal(mean = 0.0, variance = 1.0)
    τ ~ Gamma(shape = 1.0, rate = 1.0)
    y ~ Normal(mean = μ, precision = τ)
end
```

This model will fail with a `RuleNotFoundError` because there are no belief propagation message passing update rules available for this combination of distributions - only variational message passing rules exist. Even though the model looks simple, the message passing rules needed for exact inference do not exist in closed form.

## Common scenarios

You're likely to encounter this error when:

1. Using non-conjugate pairs of distributions (e.g., `Beta` prior with `Normal` likelihood with precision parameterization)
2. Working with custom distributions or factor nodes without defining all necessary update rules
3. Using complex transformations between variables that don't have defined message computations
4. Mixing different types of distributions in ways that don't have analytical solutions

## Design Philosophy

RxInfer prioritizes performance over generality in its message-passing implementation. By default, it only uses analytically derived message update rules, even in cases where numerical approximations might be possible. This design choice:

- Ensures fast and reliable inference when rules exist
- Avoids potential numerical instabilities from approximations
- Throws an error when analytical solutions don't exist

This means you may encounter `RuleNotFoundError` even in cases where approximate solutions could theoretically work. This is intentional - RxInfer will tell you explicitly when you need to consider alternative approaches rather than silently falling back to potentially slower or less reliable approximations. See the [Solutions](@ref rule-not-found-solutions) section below for more details.

## Visualizing the message passing graph

To better understand where message passing rules are needed, let's look at a simple factor graph visualization:

```mermaid
graph LR
    %% Other parts of the graph
    g1[g] -.-> x
    h1[h] -.-> z
    y -.-> g2[p]
    
    %% Main focus area
    x((x)) -.- m1[["μ<sub>x→f</sub>"]] --> f[f]
    f --> m2[["μ<sub>f→y</sub>"]] -.- y((y))
    z((z)) -.- m3[["μ<sub>z→f</sub>"]] --> f

    %% Styling
    classDef variable fill:#b3e0ff,stroke:#333,stroke-width:2px;
    classDef factor fill:#ff9999,stroke:#333,stroke-width:2px,shape:square;
    classDef otherFactor fill:#ff9999,stroke:#333,stroke-width:2px,opacity:0.3;
    classDef message fill:none,stroke:none;
    class x,y,z variable;
    class f factor;
    class g1,g2,h1 otherFactor;
    class m1,m2,m3 message;
```

In this example:
- Variables (`x`, `y`, `z`) are represented as circles
- The factor node (`f`) is represented as a square
- Messages (μ) flow along the edges between variables and factors, with subscripts indicating direction (e.g., x→f flows from x to f)
- Faded nodes (g, h) represent other parts of the factor graph that aren't relevant for this local message computation

To compute the outgoing message `f→y`, RxInfer needs:
1. Rules for how to process incoming messages `x→f` and `z→f`
2. Rules for combining these messages based on the factor `f`'s type
3. Rules for producing the outgoing message type that `y` expects

A `RuleNotFoundError` occurs when any of these rules are missing. For example, if `x` sends a `Normal` message but `f` doesn't know how to process `Normal` inputs, or if `f` can't produce the type of message that `y` expects.

## [Solutions](@id rule-not-found-solutions)

### 1. Convert to conjugate pairs

First, try to reformulate your model using conjugate prior-likelihood pairs. Conjugate pairs have analytical solutions for message passing and are well-supported in RxInfer. For example, instead of using a `Normal` likelihood with `Beta` prior on its precision, use a `Normal-Gamma` conjugate pair. See [Conjugate prior - Wikipedia](https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions) for a comprehensive list of conjugate distributions.

### 2. Check available rules

If conjugate pairs aren't suitable, verify if your combination of distributions and message types is supported. RxInfer provides many predefined rules, but not all combinations are possible. A good starting point is to check the [List of available nodes](https://reactivebayes.github.io/ReactiveMP.jl/stable/lib/nodes/#lib-predefined-nodes) section in the documentation of ReactiveMP.jl.

### 3. Create custom update rules

If you need specific message computations, you can define your own update rules. See [Creating your own custom nodes](@ref create-node) for a detailed guide on implementing custom nodes and their update rules.

### 4. Use approximations

When exact message updates aren't available, consider:

- Using simpler distribution pairs that have defined rules
- Employing approximation techniques like moment matching or the methods described in [Meta Specification](@ref user-guide-meta-specification) and [Deterministic nodes](@ref delta-node-manual)

### 5. Use variational inference

Sometimes, adding appropriate factorization constraints can help avoid problematic message computations:

```julia
constraints = @constraints begin
    q(μ, τ) = q(μ)q(τ)  # Mean-field assumption
end

result = infer(
    model = problematic_model(),
    constraints = constraints,
)
```

!!! note
    When using variational constraints, you will likely need to initialize certain messages or marginals to handle loops in the factor graph. See [Initialization](@ref initialization) for details on how to properly initialize your model.

For more details on constraints and variational inference, see:

- [Constraints Specification](@ref user-guide-constraints-specification) for a complete guide on using constraints
- [Bethe Free Energy](@ref lib-bethe-free-energy) for the mathematical background on variational inference and message passing

## Implementation details

When RxInfer encounters a missing rule, it means one of these is missing:

1. A `@rule` definition for the specific message direction and types
2. A `@marginalrule` for computing joint marginals
3. An `@average_energy` implementation for free energy computation

You can add these using the methods described in [Creating your own custom nodes](@ref create-node).

!!! note
    Not all message-passing rules have analytical solutions. In such cases, you might need to use numerical approximations or choose different model structures.

