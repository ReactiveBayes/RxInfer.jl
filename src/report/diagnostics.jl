export ReportDiagnostic

"""
    ReportDiagnostic(kind, level, subject, text)

One finding of [`RxInfer.diagnostics`](@ref): its `kind`, a `Symbol`, its `level`, `:warning` for
a part of the inference that is likely to be inaccurate or `:note` for one worth knowing, the
`subject` it is about, a variable or a node as the model writes it, and a sentence, `text`.

The kinds:
- `:laplace_not_converged` (warning): a posterior's Laplace approximation did not converge, so its
  Gaussian is the last iterate of Newton's method, its curvature floored;
- `:variational_not_converged` (warning): a posterior's variational Gaussian, [`CVI`](@ref)'s, did
  not converge;
- `:joint_not_converged` (warning): a joint the rule fallback computed by Laplace's method did not
  converge;
- `:expectation_over_messages` (warning): under belief propagation, the rule fallback took an
  expectation over another random variable's message, so its message is the variational one, an
  approximation of the belief propagation message;
- `:unprojected_messages` (note): messages Laplace found no mode for were left as log-densities,
  `LaplaceForm(nomode = :leave)`, and pulled back through the next node instead;
- `:stochastic_expectation` (note): an expectation went to a stochastic backend, so results vary
  with the random number generator;
- `:last_resort_expectation` (note): an expectation went to a backend registered as a last resort,
  at a negative priority, because no other applied.
"""
struct ReportDiagnostic
    kind::Symbol
    level::Symbol
    subject::String
    text::String
end

Base.show(io::IO, d::ReportDiagnostic) = print(io, d.level === :warning ? "warning" : "note", " (", d.kind, "): ", d.subject, ": ", d.text)

"""
    RxInfer.diagnostics(report) -> Vector{ReportDiagnostic}

The parts of an inference most likely to be inaccurate, and others worth knowing, found in the
records of an [`InferenceReport`](@ref) or of its [`ApproximationsReport`](@ref): warnings first.
Each is a [`ReportDiagnostic`](@ref); variables and nodes of the same name are folded into one,
`x[1:200]`. Empty for an inference without a report section of approximations.

```julia
result = infer(model = model(), data = data, fallback = LaplaceApproximation(), report = true)
filter(d -> d.level === :warning, RxInfer.diagnostics(result.report))
```
"""
diagnostics(report::InferenceReport) =
    :approximations in propertynames(report) ? diagnostics(report.approximations) : ReportDiagnostic[]

function diagnostics(report::ApproximationsReport)
    found = ReportDiagnostic[]
    variable_diagnostics!(found, report)
    node_diagnostics!(found, report)
    return sort!(found; by = d -> (d.level !== :warning, string(d.kind), d.subject))
end

function variable_diagnostics!(found, report::ApproximationsReport)
    notconverged = Dict{Tuple{Symbol, Symbol}, Vector{Tuple{Symbol, Any}}}()
    unprojected = Dict{Symbol, Vector{Tuple{Symbol, Any}}}()
    unprojected_count = Dict{Symbol, Int}()
    for (key, record) in report.variables
        outcome = record.marginal.outcome
        method = outcome isa NamedTuple ? get(outcome, :method, nothing) : nothing
        if record.marginal.projections > 0 && method in (:laplace, :variational) && get(outcome, :converged, true) === false
            push!(get!(notconverged, (first(key), method), Tuple{Symbol, Any}[]), key)
        end
        if record.messages.unprojected > 0
            push!(get!(unprojected, first(key), Tuple{Symbol, Any}[]), key)
            unprojected_count[first(key)] = get(unprojected_count, first(key), 0) + record.messages.unprojected
        end
    end
    for ((name, method), keys) in notconverged
        found_one = method === :laplace ?
            ReportDiagnostic(:laplace_not_converged, :warning, variables_display(keys), "Laplace did not converge: the Gaussian is at the last iterate. A prior, or `LaplaceApproximation(maxiters = …)`, may help") :
            ReportDiagnostic(:variational_not_converged, :warning, variables_display(keys), "CVI did not converge: the Gaussian is the last iterate of its bound's maximisation. `CVI(maxiters = …)` may help")
        push!(found, found_one)
    end
    for (name, keys) in unprojected
        n = unprojected_count[name]
        push!(found, ReportDiagnostic(:unprojected_messages, :note, variables_display(keys), string(n, n == 1 ? " message" : " messages", " had no mode and stayed log-densities, pulled back through the next node")))
    end
    return found
end

function node_diagnostics!(found, report::ApproximationsReport)
    joints, over, stochastic, lastresort = (Dict{Any, Vector{NodeInstance}}() for _ in 1:4)
    backends_of = Dict{Any, Set{String}}()
    targets_of = Dict{Any, Vector{Any}}()
    for instance in values(report.instances), ((kind, target), record) in instance.records
        shape = instance_shape(instance)
        record.nonconverged > 0 && push!(get!(joints, (shape, kind, target), NodeInstance[]), instance)
        if record.over_messages > 0
            push!(get!(over, shape, NodeInstance[]), instance)
            target in get!(targets_of, shape, Any[]) || push!(targets_of[shape], target)
        end
        for (Q, entry) in record.backends
            isnothing(entry.backend) && continue
            name = string(nameof(typeof(entry.backend)))
            if entry.stochastic
                push!(get!(stochastic, shape, NodeInstance[]), instance)
                push!(get!(backends_of, (:stochastic, shape), Set{String}()), name)
            elseif !isnothing(entry.priority) && entry.priority < 0
                push!(get!(lastresort, shape, NodeInstance[]), instance)
                push!(get!(backends_of, (:lastresort, shape), Set{String}()), name)
            end
        end
    end
    for ((shape, kind, target), instances) in joints
        what = kind === :energy ? "its average energy" : string("the joint of ", target_display(target))
        push!(found, ReportDiagnostic(:joint_not_converged, :warning, instances_label(unique(instances)), string("the fallback's Laplace approximation of ", what, " did not converge")))
    end
    for (shape, instances) in over
        towards = join(map(target_display, targets_of[shape]), ", ")
        push!(found, ReportDiagnostic(:expectation_over_messages, :warning, instances_label(unique(instances)), string("under BP, the messages towards ", towards, " average over another variable's message: variational, not exact BP")))
    end
    for (shape, instances) in stochastic
        push!(found, ReportDiagnostic(:stochastic_expectation, :note, instances_label(unique(instances)), string("stochastic expectations (", join(sort!(collect(backends_of[(:stochastic, shape)])), ", "), "): results vary with the RNG")))
    end
    for (shape, instances) in lastresort
        push!(found, ReportDiagnostic(:last_resort_expectation, :note, instances_label(unique(instances)), string("expectations by a last resort (", join(sort!(collect(backends_of[(:lastresort, shape)])), ", "), "): no other backend applies")))
    end
    return found
end

function diagnostics_table(found::Vector{ReportDiagnostic})
    rows = [[d.level === :warning ? "warning" : "note", d.subject, d.text] for d in found]
    return ReportTable("Diagnostics", ["", "where", "what"], rows; levels = [d.level for d in found])
end
