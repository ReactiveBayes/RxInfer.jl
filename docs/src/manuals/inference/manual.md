# [Manual inference](@id user-guide-manual-inference)

For advanced use cases it is advised to use manual inference specification. 

Manual inference specification with `RxInfer` usually consists of the same simple building blocks and designed in such a way to support both static and real-time infinite datasets:

1. Create a model with `@model` macro and get a references to random variables and data inputs
2. Subscribe to random variable posterior marginal updates 
3. Subscribe to Bethe Free Energy updates (optional)
4. Feed model with observations 
5. Unsubscribe from posterior marginal updates (optional)

It is worth to note that Step 5 is optional and in case where observations come from an infinite real-time data stream (e.g. from an external source or the internet) it may be justified to never unsubscribe and perform real-time Bayesian inference in a reactive manner as soon as data arrives.

## [Model creation](@id user-guide-manual-inference-model-creation)

During model specification stage user decides on variables of interest in a model and returns (optionally) them using a `return ...` statement. As an example consider that we have a simple hierarchical model in which the mean of a Normal distribution is represented by another Normal distribution whose mean is modelled by another Normal distribution.

```@example hierarchical-normal
using RxInfer, Distributions, Random

@model function my_model()
    m2 ~ NormalMeanVariance(0.0, 1.0)
    m1 ~ NormalMeanVariance(m2, 1.0)

    y = datavar(Float64)
    y ~ NormalMeanVariance(m1, 1.0)

    # Return variables of interests, optional
    return m1, y
end
```

And later on we may create our model with the [`create_model`](@ref) function and obtain references for variables of interests:

```@example hierarchical-normal
model, (m1, y) = create_model(my_model())
nothing #hide
```

Alternatively, it is possible to query any variable using squared brackets on `model` object:

```@example hierarchical-normal
model[:m1] # m1
model[:y]  # y
nothing #hide
```

`@model` macro also return a reference for a factor graph as its first return value. Factor graph object (named `model` in previous example) contains all information about all factor nodes in a model as well as random variables and data inputs.

## [Posterior marginal updates](@id user-guide-manual-inference-marginal-updates)

The `RxInfer` inference engine has a reactive API and operates in terms of Observables and Actors. For detailed information about these concepts we refer to [Rocket.jl documentation](https://biaslab.github.io/Rocket.jl/stable/observables/about/).

We use `getmarginal` function from `ReactiveMP` to get a posterior marginal updates observable:

```@example hierarchical-normal
m1_posterior_updates = getmarginal(m1)
nothing #hide
```

After that we can subscribe on new updates and perform some actions based on new values:

```@example hierarchical-normal
m1_posterior_subscription = subscribe!(m1_posterior_updates, (new_posterior) -> begin
    println("New posterior for m1: ", new_posterior)
end)
nothing #hide
```

Sometimes it is useful to return an array of random variables from model specification, in this case we may use `getmarginals()` function that transform an array of observables to an observable of arrays.

```julia
@model function my_model()
    ...
    m_n = randomvar(n)
    ...
    return m_n, ...
end

model, (m_n, ...) = create_model(my_model())

m_n_updates = getmarginals(m_n)
```

## [Feeding observations](@id user-guide-manual-inference-observations)

By default (without any extra factorisation constraints) model specification implies Belief Propagation message passing update rules. In case of BP algorithm `RxInfer` package computes an exact Bayesian posteriors with a single message passing iteration. To enforce Belief Propagation message passing update rule for some specific factor node user may use `where { q = FullFactorisation() }` option. Read more in [Model Specification](@ref user-guide-model-specification) section. To perform a message passing iteration we need to pass some data to all our data inputs that were created with [`datavar` function](@ref user-guide-model-specification-data-variables) during model specification.

To feed an observation for a specific data input we use `update!` function:

```@example hierarchical-normal
update!(y, 0.0)
nothing #hide
```

As you can see after we passed a single value to our data input we got a posterior marginal update from our subscription and printed it with `println` function. In case of BP if observations do not change it should not affect posterior marginal results:

```@example hierarchical-normal
update!(y, 0.0) # Observation didn't change, should result in the same posterior
nothing #hide
```

If `y` is an array of data inputs it is possible to pass an array of observation to `update!` function:

```julia
for i in 1:length(data)
    update!(y[i], data[i])
end
# is an equivalent of
update!(y, data)
```

## [Variational Message Passing](@id user-guide-manual-inference-vmp)

Variational message passing (VMP) algorithms are generated much in the same way as the belief propagation algorithm we saw in the previous section. There is a major difference though: for VMP algorithm generation we need to define the factorization properties of our approximate distribution. A common approach is to assume that all random variables of the model factorize with respect to each other. This is known as the mean field assumption. In `RxInfer`, the specification of such factorization properties is defined during model specification stage using the `where { q = ... }` syntax or with the `@constraints` macro (see [Constraints specification](@ref user-guide-constraints-specification) section for more info about the `@constraints` macro). Let's take a look at a simple example to see how it is used. In this model we want to learn the mean and precision of a Normal distribution, where the former is modelled with a Normal distribution and the latter with a Gamma.

```@example normal-estimation-vmp
using RxInfer, Distributions, Random
```

```@example normal-estimation-vmp
real_mean      = -4.0
real_precision = 0.2
rng            = MersenneTwister(1234)

n    = 100
data = rand(rng, Normal(real_mean, sqrt(inv(real_precision))), n)
nothing #hide
```

```@example normal-estimation-vmp
@model function normal_estimation(n)
    m ~ NormalMeanVariance(0.0, 10.0)
    w ~ Gamma(0.1, 10.0)

    y = datavar(Float64, n)

    for i in 1:n
        y[i] ~ NormalMeanPrecision(m, w) where { q = MeanField() }
    end

    return m, w, y
end
```

We create our model as usual, however in order to start VMP inference procedure we need to set initial posterior marginals for all random variables in the model:

```@example normal-estimation-vmp
model, (m, w, y) = create_model(normal_estimation(n))

# We use vague initial marginals
setmarginal!(m, vague(NormalMeanVariance)) 
setmarginal!(w, vague(Gamma))
nothing #hide
```

To perform a single VMP iteration it is enough to feed all data inputs with some values. To perform multiple VMP iterations we should feed our all data inputs with the same values multiple times:

```@example normal-estimation-vmp
m_marginals = []
w_marginals = []

subscriptions = subscribe!([
    (getmarginal(m), (marginal) -> push!(m_marginals, marginal)),
    (getmarginal(w), (marginal) -> push!(w_marginals, marginal)),
])

vmp_iterations = 10

for _ in 1:vmp_iterations
    update!(y, data)
end

unsubscribe!(subscriptions)
nothing #hide
```

As we process more VMP iterations, our beliefs about the possible values of `m` and `w` converge and become more confident.

```@example normal-estimation-vmp
using Plots

p1    = plot(title = "'Mean' posterior marginals")
grid1 = -6.0:0.01:4.0

for iter in [ 1, 2, 10 ]

    estimated = Normal(mean(m_marginals[iter]), std(m_marginals[iter]))
    e_pdf     = (x) -> pdf(estimated, x)

    plot!(p1, grid1, e_pdf, fill = true, opacity = 0.3, label = "Estimated mean after $iter VMP iterations")

end

plot!(p1, [ real_mean ], seriestype = :vline, label = "Real mean", color = :red4, opacity = 0.7)
```

```@example normal-estimation-vmp
p2    = plot(title = "'Precision' posterior marginals")
grid2 = 0.01:0.001:0.35

for iter in [ 2, 3, 10 ]

    estimated = Gamma(shape(w_marginals[iter]), scale(w_marginals[iter]))
    e_pdf     = (x) -> pdf(estimated, x)

    plot!(p2, grid2, e_pdf, fill = true, opacity = 0.3, label = "Estimated precision after $iter VMP iterations")

end

plot!(p2, [ real_precision ], seriestype = :vline, label = "Real precision", color = :red4, opacity = 0.7)
```

## [Computing Bethe Free Energy](@id user-guide-manual-inference-vmp-bfe)

VMP inference boils down to finding the member of a family of tractable probability distributions that is closest in KL divergence to an intractable posterior distribution. This is achieved by minimizing a quantity known as Variational Free Energy. `RxInfer` uses Bethe Free Energy approximation to the real Variational Free Energy. Free energy is particularly useful to test for convergence of the VMP iterative procedure.

The `RxInfer` package exports `score` function for an observable of free energy values:

```@example normal-estimation-vmp
fe_observable = score(model, BetheFreeEnergy())
nothing #hide
```

```@example normal-estimation-vmp
# Reset posterior marginals for `m` and `w`
setmarginal!(m, vague(NormalMeanVariance))
setmarginal!(w, vague(Gamma))

fe_values = []

fe_subscription = subscribe!(fe_observable, (v) -> push!(fe_values, v))

vmp_iterations = 10

for _ in 1:vmp_iterations
    update!(y, data)
end

unsubscribe!(fe_subscription)
```

```@example normal-estimation-vmp
plot(fe_values, label = "Bethe Free Energy", xlabel = "Iteration #")
```
