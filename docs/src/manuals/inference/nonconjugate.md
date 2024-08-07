# [Non-conjugate Inference](@id inference-nonconjugate)

The RxInfer package excels in scenarios where the model uses [conjugate priors](https://en.wikipedia.org/wiki/Conjugate_prior) for hidden states. Conjugate priors allow Bayesian inference to utilize pre-computed analytical update rules, significantly speeding up the inference process. For instance, the conjugate prior for the parameter of a `Bernoulli` distribution is the `Beta` distribution. The conjugate prior for the mean parameter of a `Normal` distribution is another `Normal` distribution, and the conjugate prior for the precision parameter of a `Normal` distribution is the `Gamma` distribution.

## Non-conjugate Structures

However, models often contain non-conjugate structures, which prevent `RxInfer` from performing efficient inference. Non-conjugate priors occur when the prior and the likelihood do not result in a posterior that belongs to the same family as the prior. This complicates the inference process because it requires approximations or numerical methods instead of simple analytical updates.

### Example Scenario

Consider the scenario where we assign the `Beta` distribution as a prior for the mean parameter of a `Normal` distribution. Let's explore what happens in this case with an example.

First, we generate some synthetic data:

```@example non-conjugacy-prior-likelihood
using Distributions, ExponentialFamily, Plots, StableRNGs

# The model will infer the hidden parameters from data
hidden_mean         = 0.2
hidden_precision    = 0.8
hidden_distribution = NormalMeanPrecision(hidden_mean, hidden_precision)

number_of_datapoints = 1000
data = rand(StableRNG(42), hidden_distribution, number_of_datapoints)

histogram(data; normalize = :pdf)
```

Next, we specify the model. Suppose we believe the data follows a `Normal` distribution, and we are confident that the mean parameter is between `0` and `1`. The `Beta` distribution is a logical choice for the prior of the mean parameter because it models a continuous variable in the range from `0` to `1`. Similarly, we assign a `Beta` prior for the precision parameter, assuming it also lies between `0` and `1`.

```@example non-conjugacy-prior-likelihood
using RxInfer

@model function non_conjugate_model(y)
   m ~ Beta(1, 1)
   p ~ Beta(1, 1)
   y .~ Normal(mean = m, precision = p)
end
```

If we attempt inference with this model, `RxInfer` will throw an error because the necessary computational rules for such a model are not available in closed form. This is due to the non-conjugate nature of the priors used.

## Addressing Non-conjugacy with ExponentialFamilyProjection

To overcome this limitation, `RxInfer` integrates with the `ExponentialFamilyProjection` package. This package re-projects non-conjugate relationships back into a member of the exponential family at the cost of some accuracy.

!!! note
    `RxInfer` supports non-conjugate inference for completeness, but be aware that inference execution times may increase significantly. This is because non-conjugate models require more complex computations, often involving sampling-based approximations.

### Specifying Constraints

The projection constraint must be specified using the `@constraints` macro. For example:

```@example non-conjugacy-prior-likelihood
using ExponentialFamilyProjection

@constraints function non_conjugate_model_constraints()
  # project variational posterior over `m` to `Beta`
  q(m) :: ProjectedTo(Beta)
  # project variational posterior over `p` to `Beta`
  q(p) :: ProjectedTo(Beta)
  # `m` and `p` are jointly independent
  q(m, p) = q(m)q(p)
end
```

These constraints specify that the posterior distribution for the hidden variable `m` must be re-projected to a `Beta` distribution to cover the region from `0` to `1`. The same applies to the variable `p`.  

!!! note 
    Note that the distribution specified in the `@constraints` does not need to match the distribution specified as a prior. For example, we could use a `Gamma` distribution as a prior and a `Beta` distribution as a posterior. The only requirement is that the support of the posterior distribution must be the same as or smaller than that of the prior.

We also assume that `m` and `p` are jointly independent with the `q(m, p) = q(m)q(p)` specification.
Dropping the assumption of joint independence would require initializing messages for `m` and `p` without guarantees of convergence.
Read more about factorization constraints in the [Constraints Specification](@ref user-guide-constraints-specification) guide.

!!! note
    The `ProjectedTo` structure is defined in the `ExponentialFamilyProjection` package. To fully explore its capabilities and hyper-parameters, we invite you to read the detailed [documentation](https://github.com/ReactiveBayes/ExponentialFamilyProjection.jl).

### Initialization

We also need to initialize the inference procedure due to the factorization constraints. Read more about initialization in the [corresponding section](@ref initialization).

```@example non-conjugacy-prior-likelihood
initialization = @initialization begin 
  q(m) = Beta(1, 1)
  q(p) = Beta(1, 1)
end
```

### Running the Inference

With everything set up, we can run the [inference](@ref user-guide-inference-execution) procedure:

```@example non-conjugacy-prior-likelihood
result = infer(
  model = non_conjugate_model(),
  data  = (y = data,),
  constraints = non_conjugate_model_constraints(),
  initialization = initialization,
  iterations = 25,
  free_energy = true
)
```

### Analyzing the Results

Let's analyze the results using the `StatsPlots` package to visualize the resulting posteriors over individual VMP iterations:

```@example non-conjugacy-prior-likelihood
using StatsPlots
using Test #hide

@test isapprox(mean(result.posteriors[:m][end]), hidden_mean, atol = 1e-1) #hide
@test isapprox(mean(result.posteriors[:p][end]), hidden_precision, atol = 1e-1) #hide

@gif for (i, q) in enumerate(zip(result.posteriors[:m], result.posteriors[:p]))
  q_m = q[1]
  q_p = q[2]

  p1 = plot(q_m, label = "Inferred `m`", fill = 0, fillalpha = 0.2)
  p1 = vline!(p1, [hidden_mean], label = "Hidden `m`")

  p2 = plot(q_p, label = "Inferred `p`", fill = 0, fillalpha = 0.2)
  p2 = vline!(p2, [hidden_precision], label = "Hidden `p`")

  plot(p1, p2; title = "Iteration $i")
end fps = 15
```

As we can see, the estimated posteriors are quite close to the actual hidden parameters used to generate our dataset. We can also verify the [Bethe Free Energy](@ref lib-bethe-free-energy) values to ensure our result has converged:

```@example non-conjugacy-prior-likelihood
@test first(result.free_energy) > last(result.free_energy) #hide
plot(result.free_energy, label = "Bethe Free Energy (per iteration)")
```

The convergence of the Bethe Free Energy indicates that the inference process has stabilized, and the model parameters have reached an optimal state.

!!! note
    The projection method uses stochastic gradient computations, which may cause fluctuations in the estimates and Bethe Free Energy performance.