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
    return postprocess_plugin(plugin, getobjective(plugin), model)
end

function GraphPPL.postprocess_plugin(::ReactiveMPFreeEnergyPlugin, objective::BetheFreeEnergy{T}, model::Model) where {T}
    skip_strategy = get_skip_strategy(objective)
    scheduler     = get_scheduler(objective)

    factor_nodes(model) do _, node
        factornode = getextra(node, :rmp_factornode)
        meta = hasextra(node, :meta) ? getextra(node, :meta) : nothing
        bfe_stream = score(__as_counting_real_type(T), FactorBoundFreeEnergy(), factornode, meta, skip_strategy, scheduler)
        setextra!(node, :bfe_stream, bfe_stream)
    end

    variable_nodes(model) do _, node
        nodeproperties = getproperties(node)::GraphPPL.VariableNodeProperties
        if is_random(nodeproperties)
            variable = getextra(node, :rmp_variable)
            bfe_stream = score(__as_counting_real_type(T), VariableBoundEntropy(), variable, skip_strategy, scheduler)
            setextra!(node, :bfe_stream, bfe_stream)
        end
    end

    return nothing
end

function score(model::ProbabilisticModel, ::BetheFreeEnergy{T}, diagnostic_checks) where {T}

    # stochastic_variables = filter(r -> !is_point_mass_form_constraint(marginal_form_constraint(r)), getrandom(model))
    # point_mass_estimates = filter(r -> is_point_mass_form_constraint(marginal_form_constraint(r)), getrandom(model))

    node_bound_free_energies = map(getfactornodes(model)) do nodedata
        nodeproperties = getproperties(nodedata)::GraphPPL.FactorNodeProperties
        stream = getextra(nodedata, :bfe_stream)
        return apply_diagnostic_check(diagnostic_checks, nodeproperties, stream)
    end

    variable_bound_entropies = map(getrandomvars(model)) do nodedata
        nodeproperties = getproperties(nodedata)::GraphPPL.VariableNodeProperties
        stream = getextra(nodedata, :bfe_stream)
        return apply_diagnostic_check(diagnostic_checks, nodeproperties, stream)
    end

    CT = __as_counting_real_type(T)

    node_bound_free_energies_sum = collectLatest(CT, CT, node_bound_free_energies, sumreduce)
    variable_bound_entropies_sum = collectLatest(CT, CT, variable_bound_entropies, sumreduce)

    degree_fn = (nodedata::GraphPPL.NodeData) -> ReactiveMP.degree(getextra(nodedata, :rmp_variable))

    data_point_entropies_n     = mapreduce(degree_fn, +, getdatavars(model), init = 0)
    constant_point_entropies_n = mapreduce(degree_fn, +, getconstantvars(model), init = 0)

    point_entropies = CountingReal(T, data_point_entropies_n + constant_point_entropies_n)

    bfe_stream = combineLatest((node_bound_free_energies_sum, variable_bound_entropies_sum), PushNew()) |> map(T, d -> float(d[1] + d[2] - point_entropies))

    return apply_diagnostic_check(diagnostic_checks, nothing, bfe_stream)
end

# Diagnostic checks 

function apply_diagnostic_check(::ObjectiveDiagnosticCheckNaNs, node::GraphPPL.FactorNodeProperties, stream)
    error_fn = let node = node
        (_) -> """
            Failed to compute node bound free energy component. The result is `NaN`. 
            Use `objective_diagnostics` keyword argument in the `inference` function to suppress this error.
            $(node)
        """
    end
    return stream |> error_if(check_isnan, error_fn)
end

function apply_diagnostic_check(::ObjectiveDiagnosticCheckNaNs, variable::GraphPPL.VariableNodeProperties, stream)
    error_fn = let variable = variable
        (_) -> """
            Failed to compute variable bound free energy component for `$(variable)` variable. The result is `NaN`. 
            Use `objective_diagnostics` keyword argument in the `inference` function to suppress this error.
        """
    end
    return stream |> error_if(check_isnan, error_fn)
end

function apply_diagnostic_check(::ObjectiveDiagnosticCheckInfs, node::GraphPPL.FactorNodeProperties, stream)
    error_fn = let node = node
        (_) -> """
            Failed to compute node bound free energy component. The result is `Inf`. 
            Use `diagnostic_checks` field in `BetheFreeEnergy` constructor or `free_energy_diagnostics` keyword argument in the `inference` function to suppress this error.
            $(node)
        """
    end
    return stream |> error_if(check_isinf, error_fn)
end

function apply_diagnostic_check(::ObjectiveDiagnosticCheckInfs, variable::GraphPPL.VariableNodeProperties, stream)
    error_fn = let variable = variable
        (_) -> """
            Failed to compute variable bound free energy component for `$(variable)` variable. The result is `Inf`. 
            Use `diagnostic_checks` field in `BetheFreeEnergy` constructor or `free_energy_diagnostics` keyword argument in the `inference` function to suppress this error.
        """
    end
    return stream |> error_if(check_isinf, error_fn)
end

## 

__as_counting_real_type(::Type{Real}) = CountingReal{<:Real}
__as_counting_real_type(::Type{T}) where {T <: Real} = CountingReal{T}