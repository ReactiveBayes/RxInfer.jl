# [Early stopping](@id manual-inference-early-stopping)


```@meta
CurrentModule = RxInfer
```

`RxInfer` supports early stopping as an opt-in callback via [`StopEarlyIterationStrategy`](@ref).
For a general overview of the callbacks system, see the [Callbacks](@ref manual-inference-callbacks) section.

## Constructors

- `StopEarlyIterationStrategy(rtol)`:
  sets `atol = 0.0`, uses the given relative tolerance.
- `StopEarlyIterationStrategy(atol, rtol)`:
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

## Example

```@example early-stopping
using RxInfer
using Test #hide

@model function iid_normal(y)
    m ~ Normal(mean = 0.0, variance = 1.0)
    tau ~ Gamma(shape = 1.0, rate = 1.0)
    y .~ Normal(mean = m, precision = tau)
end

data = (y = randn(100),)
max_iterations = 50
initialization = @initialization begin
    q(m) = NormalMeanVariance(0.0, 1.0)
    q(tau) = GammaShapeRate(1.0, 1.0)
end

result = infer(
    model = iid_normal(),
    data = data,
    constraints = MeanField(),
    initialization = initialization,
    free_energy = true,
    iterations = max_iterations,
    callbacks = (
        after_iteration = StopEarlyIterationStrategy(1e-10, 1e-3),
    )
)

@test length(result.free_energy) < max_iterations #hide
length(result.free_energy)
```

As you can see the total number of `free_energy` evaluations is less than
`max_iterations`.

