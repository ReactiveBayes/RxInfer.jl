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

### Specifying Constraints

The projection constraint must be specified using the `@constraints` macro. For example:

```@example non-conjugacy-prior-likelihood
using ExponentialFamilyProjection

@constraints function non_conjugate_model_constraints()
  q(m) :: ProjectedTo(Beta)
  q(p) :: ProjectedTo(Beta)
  # `m` and `p` are jointly independent
  q(m, p) = q(m)q(p)
end
```

These constraints specify that the posterior over the hidden variable `m` must be re-projected to the `Beta` distribution to cover the region from `0` to `1`. The same applies to the variable `p`. We also assume that `m` and `p` are jointly independent.

!!! note
    Dropping the assumption of joint independence would require initializing messages for `m` and `p` without guarantees of convergence.

To fully explore the capabilities and hyper-parameters of the `ExponentialFamilyProjection` package, we invite you to read its detailed [documentation](https://github.com/ReactiveBayes/ExponentialFamilyProjection.jl).

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
  returnvars = KeepLast(),
  iterations = 5,
  free_energy = true
)
```

### Analyzing the Results

Let's analyze the results using the `StatsPlots` package to visualize the resulting posteriors:

```@example non-conjugacy-prior-likelihood
using StatsPlots
using Test #hide

p1 = plot(result.posteriors[:m], label = "Inferred `m`", fill = 0, fillalpha = 0.2)
p1 = vline!(p1, [hidden_mean], label = "Hidden `m`")

p2 = plot(result.posteriors[:p], label = "Inferred `p`", fill = 0, fillalpha = 0.2)
p2 = vline!(p2, [hidden_precision], label = "Hidden `p`")

@test isapprox(mean(result.posteriors[:m]), hidden_mean, atol = 1e-1) #hide
@test isapprox(mean(result.posteriors[:p]), hidden_precision, atol = 1e-1) #hide

plot(p1, p2)
```

As we can see, the inference runs without any problems, and the estimated posteriors are quite close to the actual hidden parameters used to generate our dataset. We can also verify the [Bethe Free Energy](@ref lib-bethe-free-energy) values to ensure our result has converged:

```@example non-conjugacy-prior-likelihood
@test first(result.free_energy) > last(result.free_energy) #hide
plot(result.free_energy, label = "Bethe Free Energy (per iteration)")
```

The convergence of the Bethe Free Energy indicates that the inference process has stabilized, and the model parameters have reached an optimal state.

!!! note
    The projection method uses stochastic gradient computations, which may cause fluctuations in the estimates and Bethe Free Energy performance.