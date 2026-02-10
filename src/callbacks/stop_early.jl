export StopEarlyIterationStrategy

#TODO we should give an option for rtol and atol, right now it's atol

struct StopEarlyIterationStrategy
    tolerance::Float64
    start_fe_value::Float64
    fe_values::Vector{Float64}
end

function StopEarlyIterationStrategy(tolerance::Float64) 
	return StopEarlyIterationStrategy(tolerance, Inf, [])
end

function (strategy::StopEarlyIterationStrategy)(model, niteration)
    current_fe_value = 0.0
    # Subscribe on the `BetheFreeEnergy` stream but only `take(1)` value from it
    subscribe!(score(model, RxInfer.BetheFreeEnergy(Real), RxInfer.DefaultObjectiveDiagnosticChecks) |> take(1), (v) -> current_fe_value = v)
    # Take the previous value from the saved history, use the large value if the first iteration
    previous_fe_value = isempty(strategy.fe_values) ? strategy.start_fe_value : last(strategy.fe_values)
    # Save the current value in the history
    push!(strategy.fe_values, current_fe_value)
    # Stop early if the previous value is close to the current
    if abs(previous_fe_value - current_fe_value) < strategy.tolerance
        return true
    end
    # Otherwise continue
    return false
end