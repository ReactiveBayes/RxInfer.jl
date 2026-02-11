export StopEarlyIterationStrategy


"""
   StopEarlyIterationStrategy

Early-stopping criterion based on consecutive Bethe free energy (FE) values at each itertion.

Fields
- `atol::Float64`: Absolute tolerance 
- `rtol::Float64`: Relative tolerance 
- `fe_values::Vector{Float64}`: History of observed FE values (most recent is last).

The iterations stop when the current FE is approximately equal to the previous FE
under the given tolerances.
"""
struct StopEarlyIterationStrategy
    atol::Float64
    rtol::Float64
    start_fe_value::Float64
    fe_values::Vector{Float64}
end

# function StopEarlyIterationStrategy(tolerance::Float64) 
# 	return StopEarlyIterationStrategy(tolerance, Inf, Inf, [])
# end

function StopEarlyIterationStrategy(tolerance::Float64) 
	return StopEarlyIterationStrategy(0.0, tolerance, 0.0, [])
end

"""
    (strategy::StopEarlyIterationStrategy)(model) -> Bool

Returns:
- `true` if the current FE is approximately is approximately equal to the previous FE (stop iterations),
  `false` otherwise (continue iterations).
"""
function (strategy::StopEarlyIterationStrategy)(model,ntirations::Int64)
    current_fe_value = 0.0
    # Subscribe on the `BetheFreeEnergy` stream but only `take(1)` value from it
    subscribe!(score(model, RxInfer.BetheFreeEnergy(Real), RxInfer.DefaultObjectiveDiagnosticChecks) |> take(1), (v) -> current_fe_value = v)
    # Take the previous value from the saved history, use the large value if the first iteration
    previous_fe_value = isempty(strategy.fe_values) ? strategy.start_fe_value : last(strategy.fe_values)
    # Save the current value in the history
    push!(strategy.fe_values, current_fe_value)
    # Stop early if the previous value is close to the current
    return isapprox(current_fe_value, previous_fe_value; atol=strategy.atol, rtol=strategy.rtol)
end