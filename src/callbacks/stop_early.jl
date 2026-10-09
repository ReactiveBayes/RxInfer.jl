export StopEarlyIterationStrategy

"""
    StopEarlyIterationStrategy

Early-stopping criterion based on a window of consecutive Bethe free energy (FE) values.
`window = 2` preserves the consecutive-value `isapprox` criterion. Larger windows
require `maximum(values) - minimum(values) <= max(atol, rtol * maximum(abs, values))`.
History resets at iteration one of each solve or streaming observation event.

Fields
- `atol::Float64`: Absolute tolerance.
- `rtol::Float64`: Relative tolerance.
- `start_fe_value::Float64`: Initial FE reference used before the first iteration.
- `fe_values::Vector{Float64}`: History of observed FE values (most recent is last).
- `window::Int`: Number of consecutive FE values required (at least two).

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
    window::Int
end

# Preserve the previous explicit-history constructor.
StopEarlyIterationStrategy(atol, rtol, start, values) =
    StopEarlyIterationStrategy(atol, rtol, start, values, 2)

"""
    StopEarlyIterationStrategy(rtol::Real; window::Integer = 2)

Create an early-stopping strategy with `atol = 0.0` and the given `rtol`.
Uses `start_fe_value = Inf` by default.
"""
StopEarlyIterationStrategy(rtol::Real; window::Integer = 2) =
    StopEarlyIterationStrategy(0.0, rtol; window)

"""
    StopEarlyIterationStrategy(atol::Real, rtol::Real; window::Integer = 2)

Create an early-stopping strategy with explicit absolute (`atol`) and relative (`rtol`) tolerances.
Uses `start_fe_value = Inf` by default.
"""
function StopEarlyIterationStrategy(atol::Real, rtol::Real; window::Integer = 2)
    isfinite(atol) && atol >= 0 ||
        throw(ArgumentError("atol must be finite and nonnegative"))
    isfinite(rtol) && rtol >= 0 ||
        throw(ArgumentError("rtol must be finite and nonnegative"))
    window >= 2 || throw(ArgumentError("window must be at least two"))
    return StopEarlyIterationStrategy(
        Float64(atol), Float64(rtol), Inf, Float64[], Int(window)
    )
end

function stop_early_converged(strategy::StopEarlyIterationStrategy)
    values = strategy.fe_values
    if strategy.window == 2
        # Keep the original pairwise criterion, including custom initial references.
        previous =
            length(values) == 1 ? strategy.start_fe_value : values[end - 1]
        return isfinite(last(values)) &&
               isfinite(previous) &&
               isapprox(
                   last(values),
                   previous;
                   atol = strategy.atol,
                   rtol = strategy.rtol,
               )
    end
    length(values) >= strategy.window || return false
    recent = @view values[(end - strategy.window + 1):end]
    all(isfinite, recent) || return false
    # A range check catches accumulated drift even when every adjacent step is small.
    # Match isapprox's tolerance convention; absolute magnitudes also handle negative BFE.
    return maximum(recent) - minimum(recent) <=
           max(strategy.atol, strategy.rtol * maximum(abs, recent))
end

function (strategy::StopEarlyIterationStrategy)(event::AfterIterationEvent)
    # Iteration one identifies a new objective: previous observations' BFE is irrelevant.
    # Reset here so an after_iteration-only NamedTuple needs no separate reset callback.
    event.iteration == 1 && empty!(strategy.fe_values)
    current_fe_value = nothing
    # Subscribe on the `BetheFreeEnergy` stream but only `take(1)` value from it
    subscription = subscribe!(
        score(
            event.model,
            RxInfer.BetheFreeEnergy(Real),
            RxInfer.DefaultObjectiveDiagnosticChecks,
        ) |> take(1),
        (v) -> current_fe_value = v,
    )
    unsubscribe!(subscription)
    isnothing(current_fe_value) &&
        error("No Bethe free energy value available after iteration")
    # Save the current value in the history
    push!(strategy.fe_values, current_fe_value)
    # Apply the configured pairwise or full-window criterion.
    if stop_early_converged(strategy)
        event.stop_iteration = true
    end
    return nothing
end
