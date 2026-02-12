# [Early stopping](@id manual-inference-early-stopping)

```@meta
CurrentModule = RxInfer
```

`RxInfer` supports early stopping as an opt-in callback via `StopEarlyIterationStrategy`.

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

## Example

```@example early-stopping
using RxInfer

@model function iid_normal(y)
    m ~ Normal(mean = 0.0, variance = 1.0)
    tau ~ Gamma(shape = 1.0, rate = 1.0)
    y .~ Normal(mean = m, precision = tau)
end

data = (y = randn(100),)

result = infer(
    model = iid_normal(),
    data = data,
    constraints = MeanField(),
    initialization = @initialization begin
        q(m) = NormalMeanVariance(0.0, 1.0)
        q(tau) = GammaShapeRate(1.0, 1.0)
    end,
    free_energy = true,
    iterations = 50,
    callbacks = (
        after_iteration = StopEarlyIterationStrategy(1e-10, 1e-3),
    )
)

length(result.free_energy)
```
