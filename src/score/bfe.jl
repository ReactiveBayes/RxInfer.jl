export BetheFreeEnergyCheckNaNs, BetheFreeEnergyCheckInfs
export BetheFreeEnergyDefaultMarginalSkipStrategy, BetheFreeEnergyDefaultChecks
export BetheFreeEnergy

import ReactiveMP: is_point_mass_form_constraint
import ReactiveMP: CountingReal, value_isnan, value_isinf
import ReactiveMP: score, indexed_name, name

"""
    AbstractScoreObjective

Abstract type for functional objectives that can be used in the `score` function.

See also: [`score`](@ref)
"""
abstract type AbstractScoreObjective end

# Default version is differentiable
# Specialized versions like score(Float64, ...) are not differentiable, but could be faster
score(model::FactorGraphModel, objective::AbstractScoreObjective)                              = score(model, CountingReal, objective)
score(model::FactorGraphModel, ::Type{T}, objective::AbstractScoreObjective) where {T <: Real} = score(model, CountingReal{T}, objective)

# Bethe Free Energy objective

## Various check/report structures

"""
    apply_diagnostic_check(check, context, stream)

This function applies a `check` to the `stream`. Accepts optional context object for custom error messages.
"""
function apply_diagnostic_check end

"""
    BetheFreeEnergyCheckNaNs

If enabled checks that both variable and factor bound score functions in Bethe Free Energy computation do not return `NaN`s. 
Throws an error if finds `NaN`. 

See also: [`BetheFreeEnergyCheckInfs`](@ref)
"""
struct BetheFreeEnergyCheckNaNs end

function apply_diagnostic_check(::BetheFreeEnergyCheckNaNs, node::AbstractFactorNode, stream)
    error_fn = let node = node
        (_) -> """
            Failed to compute node bound free energy component. The result is `NaN`. 
            Use `diagnostic_checks` field in `BetheFreeEnergy` constructor or `free_energy_diagnostics` keyword argument in the `inference` function to suppress this error.
            $(node)
        """
    end
    return stream |> error_if(value_isnan, error_fn)
end

function apply_diagnostic_check(::BetheFreeEnergyCheckNaNs, variable::AbstractVariable, stream)
    error_fn = let variable = variable
        (_) -> """
            Failed to compute variable bound free energy component for `$(indexed_name(variable))` variable. The result is `NaN`. 
            Use `diagnostic_checks` field in `BetheFreeEnergy` constructor or `free_energy_diagnostics` keyword argument in the `inference` function to suppress this error.
        """
    end
    return stream |> error_if(value_isnan, error_fn)
end

"""
    BetheFreeEnergyCheckInfs

If enabled checks that both variable and factor bound score functions in Bethe Free Energy computation do not return `Inf`s. 
Throws an error if finds `Inf`. 

See also: [`BetheFreeEnergyCheckNaNs`](@ref)
"""
struct BetheFreeEnergyCheckInfs end

function apply_diagnostic_check(::BetheFreeEnergyCheckInfs, node::AbstractFactorNode, stream)
    error_fn = let node = node
        (_) -> """
            Failed to compute node bound free energy component. The result is `Inf`. 
            Use `diagnostic_checks` field in `BetheFreeEnergy` constructor or `free_energy_diagnostics` keyword argument in the `inference` function to suppress this error.
            $(node)
        """
    end
    return stream |> error_if(value_isinf, error_fn)
end

function apply_diagnostic_check(::BetheFreeEnergyCheckInfs, variable::AbstractVariable, stream)
    error_fn = let variable = variable
        (_) -> """
            Failed to compute variable bound free energy component for `$(indexed_name(variable))` variable. The result is `Inf`. 
            Use `diagnostic_checks` field in `BetheFreeEnergy` constructor or `free_energy_diagnostics` keyword argument in the `inference` function to suppress this error.
        """
    end
    return stream |> error_if(value_isinf, error_fn)
end

apply_diagnostic_check(::Nothing, something, stream)     = stream
apply_diagnostic_check(checks::Tuple, something, stream) = foldl((folded, check) -> apply_diagnostic_check(check, something, folded), checks; init = stream)

"""
    BetheFreeEnergy(marginal_skip_strategy, scheduler, diagnostic_checks)

Creates Bethe Free Energy values stream when passed to the `score` function. 
"""
struct BetheFreeEnergy{M, C, S} <: AbstractScoreObjective
    skip_strategy::M
    scheduler::S
    diagnostic_checks::C
end

const BetheFreeEnergyDefaultMarginalSkipStrategy = SkipInitial()
const BetheFreeEnergyDefaultChecks               = (BetheFreeEnergyCheckNaNs(), BetheFreeEnergyCheckInfs())

BetheFreeEnergy() = BetheFreeEnergy(BetheFreeEnergyDefaultMarginalSkipStrategy, AsapScheduler(), BetheFreeEnergyDefaultChecks)

get_skip_strategy(objective::BetheFreeEnergy) = objective.skip_strategy
get_scheduler(objective::BetheFreeEnergy)     = objective.scheduler

apply_diagnostic_check(objective::BetheFreeEnergy, something, stream) = apply_diagnostic_check(objective.diagnostic_checks, something, stream)

function score(model::FactorGraphModel, ::Type{T}, objective::BetheFreeEnergy) where {T <: CountingReal}
    stochastic_variables = filter(r -> !is_point_mass_form_constraint(marginal_form_constraint(r)), getrandom(model))
    point_mass_estimates = filter(r -> is_point_mass_form_constraint(marginal_form_constraint(r)), getrandom(model))

    skip_strategy = get_skip_strategy(objective)
    scheduler     = get_scheduler(objective)

    node_bound_free_energies = map(getnodes(model)) do node
        return apply_diagnostic_check(objective, node, score(T, FactorBoundFreeEnergy(), node, skip_strategy, scheduler))
    end

    variable_bound_entropies = map(stochastic_variables) do variable
        return apply_diagnostic_check(objective, variable, score(T, VariableBoundEntropy(), variable, skip_strategy, scheduler))
    end

    node_bound_free_energies_sum = collectLatest(T, T, node_bound_free_energies, sumreduce)
    variable_bound_entropies_sum = collectLatest(T, T, variable_bound_entropies, sumreduce)

    data_point_entropies_n     = mapreduce(degree, +, getdata(model), init = 0)
    constant_point_entropies_n = mapreduce(degree, +, getconstant(model), init = 0)
    form_point_entropies_n     = mapreduce(degree, +, point_mass_estimates, init = 0)

    point_entropies = CountingReal(eltype(T), data_point_entropies_n + constant_point_entropies_n + form_point_entropies_n)

    return combineLatest((node_bound_free_energies_sum, variable_bound_entropies_sum), PushNew()) |> map(eltype(T), d -> float(d[1] + d[2] - point_entropies))
end
