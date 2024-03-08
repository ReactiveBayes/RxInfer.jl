import BayesBase: CountingReal
import ReactiveMP: is_point_mass_form_constraint
import ReactiveMP: score, name

"""
    BetheFreeEnergy(marginal_skip_strategy, scheduler, diagnostic_checks)

Creates Bethe Free Energy values stream when passed to the `score` function. 
"""
struct BetheFreeEnergy{T, M, S}
    skip_strategy::M
    scheduler::S

    function BetheFreeEnergy(::Type{T}, skip_strategy::M, scheduler::S) where {T, M, S}
        return new{T, M, S}(skip_strategy, scheduler)
    end
end

const BetheFreeEnergyDefaultMarginalSkipStrategy = SkipInitial()
const BetheFreeEnergyDefaultScheduler = AsapScheduler()

BetheFreeEnergy(::Type{T}) where {T} = BetheFreeEnergy(T, BetheFreeEnergyDefaultMarginalSkipStrategy, BetheFreeEnergyDefaultScheduler)

get_skip_strategy(objective::BetheFreeEnergy) = objective.skip_strategy
get_scheduler(objective::BetheFreeEnergy)     = objective.scheduler

struct ReactiveMPFreeEnergyPlugin{O}
    objective::O
end

getobjective(plugin::ReactiveMPFreeEnergyPlugin) = plugin.objective

GraphPPL.plugin_type(::ReactiveMPFreeEnergyPlugin) = FactorAndVariableNodesPlugin()

function GraphPPL.preprocess_plugin(::ReactiveMPFreeEnergyPlugin, ::Model, ::Context, label::NodeLabel, nodedata::NodeData, ::NodeCreationOptions)
    return label, nodedata
end

function GraphPPL.postprocess_plugin(plugin::ReactiveMPFreeEnergyPlugin, model::Model)
    return nothing
end

function score(model::ProbabilisticModel, objective::BetheFreeEnergy, diagnostic_checks)
    return of(0.0)

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

    bfe_stream = combineLatest((node_bound_free_energies_sum, variable_bound_entropies_sum), PushNew()) |> map(eltype(T), d -> float(d[1] + d[2] - point_entropies))

    return apply_diagnostic_check(objective, CountingReal, bfe_stream)
end

# Diagnostic checks 

function apply_diagnostic_check(::ObjectiveDiagnosticCheckNaNs, objective::BetheFreeEnergy, node::AbstractFactorNode, stream)
    error_fn = let node = node
        (_) -> """
            Failed to compute node bound free energy component. The result is `NaN`. 
            Use `objective_diagnostics` keyword argument in the `inference` function to suppress this error.
            $(node)
        """
    end
    return stream |> error_if(check_isnan, error_fn)
end

function apply_diagnostic_check(::ObjectiveDiagnosticCheckNaNs, objective::BetheFreeEnergy, variable::AbstractVariable, stream)
    error_fn = let variable = variable
        (_) -> """
            Failed to compute variable bound free energy component for `$(variable)` variable. The result is `NaN`. 
            Use `objective_diagnostics` keyword argument in the `inference` function to suppress this error.
        """
    end
    return stream |> error_if(check_isnan, error_fn)
end

function apply_diagnostic_check(::ObjectiveDiagnosticCheckInfs, objective::BetheFreeEnergy, node::AbstractFactorNode, stream)
    error_fn = let node = node
        (_) -> """
            Failed to compute node bound free energy component. The result is `Inf`. 
            Use `diagnostic_checks` field in `BetheFreeEnergy` constructor or `free_energy_diagnostics` keyword argument in the `inference` function to suppress this error.
            $(node)
        """
    end
    return stream |> error_if(check_isinf, error_fn)
end

function apply_diagnostic_check(::ObjectiveDiagnosticCheckInfs, objective::BetheFreeEnergy, variable::AbstractVariable, stream)
    error_fn = let variable = variable
        (_) -> """
            Failed to compute variable bound free energy component for `$(variable)` variable. The result is `Inf`. 
            Use `diagnostic_checks` field in `BetheFreeEnergy` constructor or `free_energy_diagnostics` keyword argument in the `inference` function to suppress this error.
        """
    end
    return stream |> error_if(check_isinf, error_fn)
end
