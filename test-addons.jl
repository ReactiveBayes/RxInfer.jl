using RxInfer, Random, Plots
import ReactiveMP: getaddons, LoggerPipelineStage
import ReactiveMP: AbstractPipelineStage, apply_pipeline_stage, functionalform, tap, as_message
using JLD2

struct MessageProbePipelineStage{T} <: AbstractPipelineStage
    output::T
    prefix::String
end

MessageProbePipelineStage() = MessageProbePipelineStage(Core.stdout, "Probe")
MessageProbePipelineStage(output::IO) = MessageProbePipelineStage(output, "Probe")
MessageProbePipelineStage(prefix::String) = MessageProbePipelineStage(Core.stdout, prefix)

# Helper print function
function probe_println(probe::MessageProbePipelineStage, msg)
    println(probe.output, "[", probe.prefix, "]: ", msg)
end

# Main hook: applied to all outgoing messages from factor nodes
function ReactiveMP.apply_pipeline_stage(probe::MessageProbePipelineStage, factornode, tag::Val{T}, stream) where {T}
    stream |> tap(v -> begin
        m = as_message(v)  # realize DeferredMessage
        probe_println(probe, "[$(functionalform(factornode))][$T]: $(m.addons)")
    end)
end

#function ReactiveMP.apply_pipeline_stage(probe::MessageProbePipelineStage, factornode, tag::Tuple{Val{T}, Int}, stream) where {T}
#    stream |> tap(v -> begin
#        m = as_message(v)
#        m = typeof(m)
#        probe_println(probe, "[$(functionalform(factornode))][$T:$(tag[2])]: $(m)")
#    end)
#end
#

mutable struct CollectingProbePipelineStage <: AbstractPipelineStage
    data::Vector{Any}  # will hold all records
end

CollectingProbePipelineStage() = CollectingProbePipelineStage(Any[])

function ReactiveMP.apply_pipeline_stage(probe::CollectingProbePipelineStage, factornode, tag::Val{T}, stream) where {T}
    stream |> tap(v -> begin
        m = as_message(v)
        try
            addon = only(m.addons).memory  # AddonMemory(...)
            inputs = addon.marginals
            result = addon.result
            push!(probe.data, (
                node = functionalform(factornode),
                interface = T,
                inputs = inputs,
                result = result,
            ))
        catch err
            @warn "Could not collect message info" exception=(err, catch_backtrace())
        end
    end)
end
#probe = MessageProbePipelineStage()
#probe = CollectingProbePipelineStage()

# A callback that will be called every time after a variational iteration finishes
#function after_iteration_callback(model, iteration)
#    println("Iteration ", iteration, " has been finished")
#end

# A callback that will be called every time a posterior is updated
#function on_marginal_update_callback(model, variable_name, posterior)
#    println("Latent variable ", variable_name, " has been updated. Estimated mean is ", mean(posterior), " with standard deviation ", std(posterior))
#end

mutable struct IterationAwareProbe <: AbstractPipelineStage
    id::Symbol
    current_iter::Int
    data::Dict{Int, Vector{Any}}
end

IterationAwareProbe(id::Symbol) = IterationAwareProbe(id, 0, Dict{Int, Vector{Any}}())

function next_iteration!(probe::IterationAwareProbe, iter)
    probe.current_iter = iter
    probe.data[iter] = Any[]
end

function ReactiveMP.apply_pipeline_stage(probe::IterationAwareProbe, factornode, tag::Val{T}, stream) where {T}
    stream |> tap(v -> begin
        m = as_message(v)
        addon = only(m.addons).memory
        push!(get!(probe.data, probe.current_iter, Any[]),
            (
                node = functionalform(factornode),
                interface = T,
                inputs = addon.marginals,
                result = addon.result
            )
        )
    end)
end

const node_probes = Dict{Symbol, IterationAwareProbe}()

function make_probe(node_name::Symbol)
    probe = IterationAwareProbe(node_name)
    node_probes[node_name] = probe
    return probe
end

@model function iid_normal(y)
    μ  ~ Normal(mean = 0.0, variance = 100.0) where { pipeline = make_probe(:μ) }
    γ  ~ Gamma(shape = 1.0, rate = 1.0) where { pipeline = make_probe(:γ) }
    y .~ Normal(mean = μ, precision = γ) where { pipeline = make_probe(:y) }
end

# A callback that will be called every time before a variational iteration starts
function before_iteration_callback(model, iteration)
    for probe in values(node_probes)
        next_iteration!(probe, iteration)
    end
end

init = @initialization begin
    q(μ) = vague(NormalMeanVariance)
end

dataset = rand(NormalMeanPrecision(3.1415, 30.0), 100)

result = infer(
    model = iid_normal(),
    data  = (y = dataset, ),
    constraints = MeanField(),
    initialization = init,
    addons = (AddonMemory(),),
    iterations = 5,
    callbacks = (
        #on_marginal_update = on_marginal_update_callback,
        before_iteration   = before_iteration_callback,
        #after_iteration    = after_iteration_callback
    )
)
for (name, probe) in node_probes
    filename = "plotting/results/probe_$(name).jld2"
    @save filename probe
end
#getaddons(result.posteriors)
