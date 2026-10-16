# [Early stopping](@id manual-inference-early-stopping)


```@meta
CurrentModule = RxInfer
```

`RxInfer` supports early stopping as an opt-in callback via [`StopEarlyIterationStrategy`](@ref).
For a general overview of the callbacks system, see the [Callbacks](@ref manual-inference-callbacks) section.

## Constructors

- `StopEarlyIterationStrategy(rtol; window = 2)`:
  sets `atol = 0.0`, uses the given relative tolerance.
- `StopEarlyIterationStrategy(atol, rtol; window = 2)`:
  sets both absolute and relative tolerances.
- Both constructors use `start_fe_value = Inf` for the initial comparison value.

```@docs
RxInfer.StopEarlyIterationStrategy
RxInfer.StopEarlyIterationStrategy(rtol::Real)
RxInfer.StopEarlyIterationStrategy(atol::Real, rtol::Real)
```

## Early stopping mechanism

The [`BeforeIterationEvent`](@ref) and [`AfterIterationEvent`](@ref) carry a mutable `stop_iteration::Bool` field (default `false`).
Any callback can set `event.stop_iteration = true` to signal the inference engine to stop iterating.
The [`StopEarlyIterationStrategy`](@ref) uses this mechanism internally — when the free energy has converged, it sets `event.stop_iteration = true`.

Check out more about callbacks for static inference [here](@ref manual-static-inference-callbacks).

Note that in this case we still have to specify the `iterations`, which
in the case of early stopping specifies _maximum_ number of iterations.

The default `window = 2` compares consecutive values with `isapprox`, preserving
the previous behavior. For example, `window = 4` requires four finite BFE values
whose full range satisfies
`maximum(values) - minimum(values) <= max(atol, rtol * maximum(abs, values))`.
This prevents several individually small changes from accumulating into a larger
drift within the window. Four values span three iteration-to-iteration changes.

The same `before_iteration` and `after_iteration` callbacks are supported in
streaming inference. A stop flag ends only the current observation event's solve;
posterior publication, history saving, and processing of future events continue.
The strategy resets its BFE history at iteration one of every new solve/event, so
values from different observations are never compared. Streaming BFE histories
contain only executed iterations; the per-iteration average includes only events
that reached that iteration. A stop before the first iteration produces no tick
or history entry for that observation event.

Use a separate strategy instance for each concurrently running inference engine.
BFE stability is a stopping heuristic, not a guarantee of posterior accuracy.

## Batch example: early stopping and convergence windows

Consider five noisy sensor readings with an unknown mean and precision. First,
we demonstrate how an early-stopping callback saves iterations compared with a
fixed budget of 50. We also compare the default `window = 2` with `window = 4`,
using the same BFE tolerance, to see how the window affects when inference stops
and how closely the estimated mean agrees with the full run.

```@example early-stopping
using RxInfer
using Test #hide

@model function iid_normal(y, n)
    m ~ Normal(mean = 0.0, variance = 1.0)
    tau ~ Gamma(shape = 1.0, rate = 1.0)
    for i in 1:n
        y[i] ~ Normal(mean = m, precision = tau)
    end
end

data = (y = [3.2, 3.7, 4.0, 4.4, 4.7],)
max_iterations = 50
initialization = @initialization begin
    q(m) = NormalMeanVariance(0.0, 1.0)
    q(tau) = GammaShapeRate(1.0, 1.0)
end

function fit_batch(window)
    return infer(
        model = iid_normal(n = length(data.y)),
        data = data,
        constraints = MeanField(),
        initialization = initialization,
        free_energy = true, # The stopping callback monitors BFE.
        iterations = max_iterations, # With early stopping, this is an upper limit.
        returnvars = (m = KeepLast(),),
        callbacks = isnothing(window) ? nothing : (
            after_iteration = StopEarlyIterationStrategy(0.0, 0.01; window),
        ),
    )
end

# No callback: execute the full iteration budget as a reference.
reference = fit_batch(nothing)
# Early stopping: compare consecutive BFE values (2) with a four-value window (4).
results = [fit_batch(window) for window in (2, 4)]
mean_errors = [abs(mean(r.posteriors[:m]) - mean(reference.posteriors[:m])) for r in results]
@test length(results[1].free_energy) < length(results[2].free_energy) < max_iterations #hide
@test mean_errors[2] < mean_errors[1] / 5 #hide
pairwise_bfe = results[1].free_energy #hide
@test abs(pairwise_bfe[end] - pairwise_bfe[end - 1]) <= 0.01 * maximum(abs, pairwise_bfe[end-1:end]) #hide
@test maximum(pairwise_bfe[end-3:end]) - minimum(pairwise_bfe[end-3:end]) > 0.01 * maximum(abs, pairwise_bfe[end-3:end]) #hide
[(; window, iterations = length(r.free_energy),
    saved_iterations = length(reference.free_energy) - length(r.free_energy),
    mean_error = error)
 for (window, r, error) in zip((2, 4), results, mean_errors)]
```

Both callbacks stop before the 50-iteration limit. `saved_iterations` counts the
rounds of message updates avoided, while `mean_error` measures the difference
from the full run's estimated mean.

With `window = 2`, two adjacent BFE values pass the 1% tolerance while the BFE
is still drifting across the last four iterations. `window = 4` waits for a
flatter plateau and gives a mean closer to the reference, while still saving
iterations. The readings are noisy, but inference here is deterministic; this
example demonstrates accumulated BFE drift. A larger window remains a heuristic,
so check the posterior quantities that matter for your application.

## Streaming example: stopping each observation separately

Suppose a sensor sends batches of noisy readings, and we estimate the mean and
precision of each batch using the model and initialization above. Streaming early
stopping avoids spending the full 50 iterations on every batch. We use the default
`window = 2` here. The model's explicit size `n` lets streaming inference construct
the graph before readings arrive.

```@example early-stopping
batches = [offset .+ [-0.8, -0.3, 0.0, 0.4, 0.7] for offset in (0.0, 0.0, 2.0)]
engine = infer(
    model = iid_normal(n = 5),
    data = (y = batches,),
    autoupdates = RxInfer.EmptyAutoUpdateSpecification,
    constraints = MeanField(),
    initialization = initialization,
    iterations = max_iterations,
    free_energy = true,
    callbacks = (after_iteration = StopEarlyIterationStrategy(1e-10, 1e-8),),
    keephistory = length(batches),
    historyvars = (m = KeepEach(),),
)

iterations_per_batch = length.(engine.history[:m])
@test length(iterations_per_batch) == length(batches) #hide
@test all(n -> 2 <= n < max_iterations, iterations_per_batch) #hide
@test mean(last(engine.history[:m][end])) > mean(last(engine.history[:m][1])) #hide
(; iterations_per_batch,
   saved_iterations = length(batches) * max_iterations - sum(iterations_per_batch))
```

The empty auto-update specification selects streaming inference without carrying
posterior parameters into the next batch's prior. The second batch repeats the
first, but still needs its own two-value comparison: the strategy resets its BFE
history for each batch. Stopping one batch also leaves the engine ready to process
the third batch, whose mean has shifted. `KeepEach()` records only the iterations
actually executed; without the stopping callback, each batch would use all 50.

