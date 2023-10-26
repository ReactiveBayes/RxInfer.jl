# [Constraints Specification](@id user-guide-constraints-specification)
When conducting Variational Inference, `RxInfer.jl` minimizes the Bethe approximation to the Variational Free Energy, the Bethe Free Energy (BFE). More information on the BFE and it's implications can be found [here](https://rxinfer.ml). For the sake of this article, we will assume basic familiarity with the BFE and merely repeat its definition here.

The Bethe Free Energy, given a factorized model `` f(\mathbf{s}) = \prod_{a \in \mathcal{V}}f_a(\mathbf{s}_a) `` and a normalized probability distribution `` q(s) ``, is defined as follows:

```math
F[q, f] = \sum_{a \in \mathcal{V}} \int q_a(\mathbf{s}_a)\log \frac{q_a(\mathbf{s}_a)}{f_a(\mathbf{s}_a)}d\mathbf{s}_a + \sum_{i \in \mathcal{E}}\int q_i(s_i) \log \frac{1}{q_i(s_i)} ds_i
```
such that the factorized beliefs

```math
q(\mathbf{s}) = \prod_{a \in \mathcal{V}} q_a(\mathbf{s}_a) \prod_{i \in \mathcal{E}} q_i(s_i)^{-1}
```
satisfy the following constraints:

```math
\int q_a(\mathbf{s}_a)d\mathbf{s}_a = 1 \qquad \forall a \in \mathcal{V} \\
\int q_a(\mathbf{s}_a)d\mathbf{s}_{a \setminus i} = q_i(s_i) \qquad \forall a \in \mathcal{V}, i \in \mathcal{E}
```

In `RxInfer`, we use the `@model` macro to specify our generative model ``f(\mathbf{s})``. In [csenoz2021variational](@cite), it has been shown that by including additional constraints on the variational posterior ``q(\mathbf{s})``, we can obtain many well known inference algorithms. Therefore, to `RxInfer` exports the `@constraints` macro to define additional constraints on the variational posterior, that will be passed to the inference engine.
## `@constraints` syntax
Additional constraints on the variational posterior can be defined with the `@constraints` macro. We will explain this syntax using an extended version of [the Bayesian Coin Toss example](https://biaslab.github.io/RxInfer.jl/stable/examples/basic_examples/Coin%20Toss%20Model/), where we put a non-conjugate prior on the parameters of the prior on ``\theta``:

```@example manual_constraints
using RxInfer

@model function coin_model(n)
    y = datavar(Float64, n)

    α ~ Gamma(1, 1)
    β ~ Gamma(1, 1)

    θ ~ Beta(α, β)
    for i in 1:n
        y[i] ~ Bernoulli(θ)
    end
end
```
With `@constraints`, we can design additional constraints to impose on this model. The `@constraints` macro accepts a list of factorization- and functional form constraints.
### Factorization Constraints
A factorization constraint imposes additional independence constraints on factorized beliefs in the variational posterior:
```math
q_b(\mathbf{s}_b) = \prod_{n \in l(b)} q_b^n(\mathbf{s}_b^n)
```
for any partition ``l(b)`` of ``\mathbf{s}_b`` for a node ``b``. 

An example of a factorization constraint is the mean-field approximation, where we assume that ``q_b(\mathbf{s}_b)`` decomposes into individual, independent components. In our Bayesian Coin toss example, we might want to assume independence in the variational posterior between `θ`,  `α` and `β`. We can do this with the following constraints:

```@example manual_constraints
@constraints begin
    q(θ, α, β) = q(θ)q(α)q(β)
end
```
Note that this specifies a mean-field constraint on the `Beta` node in our model. The mean-field constraint enjoys a special place in our syntax, and we can also specify it using `MeanField()`:
    
```julia
@constraints begin
    q(θ, α, β) = MeanField()
end
```
Here, we see the general syntax of adding factorization constraints to our model. We use the `q(__variables__) = __factorization__` pattern to specify which constraint to apply. `RxInfer` will derive in which nodes to apply this constraint. Note that, since `RxInfer` explicitly makes the Bethe assumption, that in order to assume independence between two variables in the variational posterior, they must share a factor node neighbor, as otherwise there does not exist a `q_b(\mathb{s}_b)` in which both variables occur.

Factorization constraints over arrays of variables can also be specified with the same syntax. For example, if we want to add a factorization constraint between the `y` variables in our model (even though they are not connected in the factor graph), we can do so with the following syntax:

```@example manual_constraints
@constraints begin
    q(y) = q(y[begin])..q(y[end])
end
```

Joining this with our previous mean-field assumption over `θ`,  `α` and `β`, this gives us the following constraints:

```julia
@constraints begin
    q(θ, α, β) = MeanField()
    q(y) = q(y[begin])..q(y[end])
end
```
!!! note 
    `@constraints` macro does not support matrix-based collections of variables. E.g. it is not possible to write `q(x[begin, begin])..q(x[end, end])` for a matrix `x`.
### Functional Form Constraints
Functional Form Constraints restrict the functional form of the variational posterior for a specific variable. For example, we might want to restrict the variational posterior of `θ` to be a `PointMass` distribution, such that we get the MAP estimate for `θ` as our variational posterior. We can add this constraint with the following syntax:

```julia
@constraints begin
    q(θ, α, β) = MeanField()
    q(y) = q(y[begin])..q(y[end])
    q(θ) :: PointMass
end
```
This indicates that the resulting marginal of the variable (or array of variables) named `x` must be approximated with a `PointMass` object. Message passing based algorithms compute posterior marginals as a normalized product of two colliding messages on corresponding edges of a factor graph. Mathematically, `q(x)::PointMass` reads as:

```math
\mathrm{approximate~} q(x) = \frac{\overrightarrow{\mu}(x)\overleftarrow{\mu}(x)}{\int \overrightarrow{\mu}(x)\overleftarrow{\mu}(x) \mathrm{d}x}\mathrm{~as~PointMass}
```
Similar to factorization constraints, functional form constraints are parsed by `RxInfer` internally and will resolve constraints for array- or matrix-based variables. 
## Constraints in submodels
`RxInfer` allows you to define your generative model hierarchically, using previously defined `@model` modules as submodels in larger models. Because of this, users need to specify their constraints hierarchically as well to avoid ambiguities. Consider the following example:

```julia
@model function inner_inner(τ, y)
    y ~ Normal(τ[1], τ[2])
end

@model function inner(θ, α)
    β ~ Normal(0, 1)
    α ~ Gamma(β, 1)
    α ~ inner_inner(τ = θ)
end

@model function outer()
    local w
    for i = 1:5
        w[i] ~ inner(θ = Gamma(1, 1))
    end
    y ~ inner(θ = w[2:3])
end
```

To access the variables in the submodels, we use the `for q in __submodel__` syntax, which will allow us to specify constraints over variables in the context of an inner submodel:

```julia
@constraints begin
    for q in inner
        q(α) :: PointMass
        q(α, β) = q(a)q(b)
    end
end
```

Similarly, we can specify constraints over variables in the context of the innermost submodel by using the `for q in __submodel__` syntax twice:

```julia
@constraints begin
    for q in inner
        for q in inner_inner
            q(y, τ) = q(y)q(τ[1])q(τ[2])
        end
        q(α) :: PointMass
        q(α, β) = q(a)q(b)
    end
end
```

The `for q in __submodel__` applies the constraints specified in this code block to all instances of `__submodel__` in the current context. If we want to apply constraints to a specific instance of a submodel, we can use the `for q in (__submodel__, __identifier__)` syntax, where `__identifier__` is a counter integer. For example, if we want to specify constraints on the first instance of `inner` in our `outer` model, we can do so with the following syntax:

```julia
@constraints begin
    for q in (inner, 1)
        q(α) :: PointMass
        q(α, β) = q(a)q(b)
    end
end
```

Factorization constraints specified in a context propagate to their child submodels. This means that we can specify factorization constraints over variables where the factor node that connects the two are in a submodel, without having to specify the factorization constraint in the submodel itself. For example, if we want to specify a factorization constraint between `w[2]` and `w[3]` in our `outer` model, we can specify it in the context of `outer`, and `RxInfer` will recognize that these variables are connected through the `Normal` node in the `inner_inner` submodel:

```julia
@constraints begin
    q(w) = q(w[begin])..q(w[end])
end
```
