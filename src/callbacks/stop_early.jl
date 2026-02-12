export StopEarlyIterationStrategy

"""
    StopEarlyIterationStrategy

Early-stopping criterion based on consecutive Bethe free energy (FE) values.

Fields
- `atol::Float64`: Absolute tolerance.
- `rtol::Float64`: Relative tolerance.
- `start_fe_value::Float64`: Initial FE reference used before the first iteration.
- `fe_values::Vector{Float64}`: History of observed FE values (most recent is last).

Constructors
- `StopEarlyIterationStrategy(rtol)`: uses `atol = 0.0`, custom `rtol`.
- `StopEarlyIterationStrategy(atol, rtol)`: custom absolute and relative tolerances.

Both constructors use `start_fe_value = Inf` by default to avoid immediate stopping on the first iteration.
"""
struct StopEarlyIterationStrategy
    atol::Float64
    rtol::Float64
    start_fe_value::Float64
    fe_values::Vector{Float64}
end

"""
    StopEarlyIterationStrategy(rtol::Real)

Create an early-stopping strategy with `atol = 0.0` and the given `rtol`.
Uses `start_fe_value = Inf` by default.
"""
StopEarlyIterationStrategy(rtol::Real) = StopEarlyIterationStrategy(0.0, rtol)

"""
    StopEarlyIterationStrategy(atol::Real, rtol::Real)

Create an early-stopping strategy with explicit absolute (`atol`) and relative (`rtol`) tolerances.
Uses `start_fe_value = Inf` by default.
"""
StopEarlyIterationStrategy(atol::Real, rtol::Real) = StopEarlyIterationStrategy(Float64(atol), Float64(rtol), Inf, Float64[])

function (strategy::StopEarlyIterationStrategy)(model, iteration::Int)
    current_fe_value = 0.0
    # Subscribe on the `BetheFreeEnergy` stream but only `take(1)` value from it
    subscribe!(score(model, RxInfer.BetheFreeEnergy(Real), RxInfer.DefaultObjectiveDiagnosticChecks) |> take(1), (v) -> current_fe_value = v)
    # Take the previous value from the saved history, use the large value if the first iteration
    previous_fe_value = isempty(strategy.fe_values) ? strategy.start_fe_value : last(strategy.fe_values)
    # Save the current value in the history
    push!(strategy.fe_values, current_fe_value)
    # Stop early if the previous value is close to the current
    return isapprox(current_fe_value, previous_fe_value; atol = strategy.atol, rtol = strategy.rtol)
end
