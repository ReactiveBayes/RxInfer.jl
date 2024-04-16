import BayesBase: CountingReal
import ReactiveMP: score

"""
    BetheFreeEnergy(skip_strategy, scheduler)

Implements a reactive stream for Bethe Free Energy values. 
Must be used in combination with the `score` function of `ReactiveMP.jl`. 

# Arguments
- `::Type{T}`: a type of the counting real number, e.g. `Float64`. Set to `Real` by default, otherwise the inference procedure is not automatically differentiable.
- `skip_strategy`: a strategy that defines which posterior marginals to skip, e.g. `SkipInitial()`.
- `scheduler`: a scheduler for the underlying stream, e.g. `AsapScheduler()`.
"""
struct BetheFreeEnergy{T, M, S}
    skip_strategy::M
    scheduler::S

    function BetheFreeEnergy(::Type{T}, skip_strategy::M, scheduler::S) where {T, M, S}
        return new{T, M, S}(skip_strategy, scheduler)
    end
end

"""
Default marginal skip strategy for the Bethe Free Energy objective. 
"""
const BetheFreeEnergyDefaultMarginalSkipStrategy = SkipInitial()

"""
Default scheduler for the Bethe Free Energy objective.
"""
const BetheFreeEnergyDefaultScheduler = AsapScheduler()

BetheFreeEnergy(::Type{T}) where {T} = BetheFreeEnergy(T, BetheFreeEnergyDefaultMarginalSkipStrategy, BetheFreeEnergyDefaultScheduler)

get_skip_strategy(objective::BetheFreeEnergy) = objective.skip_strategy
get_scheduler(objective::BetheFreeEnergy)     = objective.scheduler

"""
A plugin for GraphPPL graph engine that adds the Bethe Free Energy objective computation to the nodes of the model.
"""
struct ReactiveMPFreeEnergyPlugin{O}
    objective::O
end

getobjective(plugin::ReactiveMPFreeEnergyPlugin) = plugin.objective

const ReactiveMPExtraBetheFreeEnergyStreamKey = GraphPPL.NodeDataExtraKey{:bfe_stream, Any}()

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
        factornode = getextra(node, ReactiveMPExtraFactorNodeKey)
        metadata = getextra(node, GraphPPL.MetaExtraKey, nothing)
        bfe_stream = score(__as_counting_real_type(T), FactorBoundFreeEnergy(), factornode, metadata, skip_strategy, scheduler)
        setextra!(node, ReactiveMPExtraBetheFreeEnergyStreamKey, bfe_stream)
    end

    variable_nodes(model) do _, node
        nodeproperties = getproperties(node)::GraphPPL.VariableNodeProperties
        if is_random(nodeproperties)
            variable = getextra(node, ReactiveMPExtraVariableKey)
            bfe_stream = score(__as_counting_real_type(T), VariableBoundEntropy(), variable, skip_strategy, scheduler)
            setextra!(node, ReactiveMPExtraBetheFreeEnergyStreamKey, bfe_stream)
        end
    end

    return nothing
end

function score(model::ProbabilisticModel, ::BetheFreeEnergy{T}, diagnostic_checks) where {T}
    node_bound_free_energies = map(getfactornodes(model)) do nodedata
        nodeproperties = getproperties(nodedata)::GraphPPL.FactorNodeProperties
        stream = getextra(nodedata, ReactiveMPExtraBetheFreeEnergyStreamKey)
        return apply_diagnostic_check(diagnostic_checks, nodeproperties, stream)
    end

    variable_bound_entropies = map(getrandomvars(model)) do nodedata
        nodeproperties = getproperties(nodedata)::GraphPPL.VariableNodeProperties
        stream = getextra(nodedata, ReactiveMPExtraBetheFreeEnergyStreamKey)
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
