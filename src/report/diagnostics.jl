export ReportDiagnostic

"""
    ReportDiagnostic(kind, level, subject, text)

One finding of [`RxInfer.diagnostics`](@ref): its `kind`, a `Symbol`, its `level`, `:warning` for
a part of the inference that is likely to be inaccurate or `:note` for one worth knowing, the
`subject` it is about, a variable or a node as the model writes it, and a sentence, `text`.

The kinds:
- `:laplace_not_converged` (warning): a posterior's Laplace approximation did not converge, so its
  Gaussian is the last iterate of Newton's method, its curvature floored;
- `:variational_not_converged` (warning): a posterior's variational Gaussian, the projection of a
  belief without sites under [`NaturalGradientPropagation`](@ref), did not converge;
- `:moment_matching_not_converged` (warning): a posterior's moment matching, under
  [`ExpectationPropagation`](@ref), did not converge, so its Gaussian is the one its quadrature
  points were placed on;
- `:sites_not_converged` (warning): under [`LaplacePropagation`](@ref) or
  [`ExpectationPropagation`](@ref), the projections of some of a variable's sites did not converge;
  under [`NaturalGradientPropagation`](@ref), the natural-gradient steps of some of its sites had
  not settled at the last sweep, so they are not yet at their targets;
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
result = infer(model = model(), data = data, fallback = LocalLaplace(), report = true)
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
    sites = Dict{Symbol, Vector{Tuple{Symbol, Any}}}()
    sites_count = Dict{Symbol, Tuple{Int, Int, Any}}()
    unprojected = Dict{Symbol, Vector{Tuple{Symbol, Any}}}()
    unprojected_count = Dict{Symbol, Int}()
    for (key, record) in report.variables
        outcome = record.marginal.outcome
        method = outcome isa NamedTuple ? get(outcome, :method, nothing) : nothing
        if record.marginal.projections > 0 && method in (:laplace, :variational, :moment_matching) && get(outcome, :converged, true) === false
            push!(get!(notconverged, (first(key), method), Tuple{Symbol, Any}[]), key)
        end
        if record.sites.nonconverged > 0
            push!(get!(sites, first(key), Tuple{Symbol, Any}[]), key)
            (failed, total, _) = get(sites_count, first(key), (0, 0, nothing))
            method = get(record.sites.outcome, :method, nothing)
            computed = method === :natural_gradient ? length(record.settled) : record.sites.projections
            sites_count[first(key)] = (failed + record.sites.nonconverged, total + computed, method)
        end
        if record.messages.unprojected > 0
            push!(get!(unprojected, first(key), Tuple{Symbol, Any}[]), key)
            unprojected_count[first(key)] = get(unprojected_count, first(key), 0) + record.messages.unprojected
        end
    end
    hint = maxiters_hint(report.scheme)
    for ((name, method), keys) in notconverged
        found_one = if method === :laplace
            ReportDiagnostic(:laplace_not_converged, :warning, variables_display(keys), string("Laplace did not converge: the Gaussian is at the last iterate. A prior, or ", hint, ", may help"))
        elseif method === :variational
            ReportDiagnostic(:variational_not_converged, :warning, variables_display(keys), string("the variational Gaussian did not converge: it is the last iterate of its bound's maximization. ", uppercasefirst(hint), " may help"))
        else
            ReportDiagnostic(:moment_matching_not_converged, :warning, variables_display(keys), string("moment matching did not converge: the Gaussian is the one its points were placed on. A prior, or ", hint, ", may help"))
        end
        push!(found, found_one)
    end
    for (name, keys) in sites
        (failed, total, method) = sites_count[name]
        if method === :natural_gradient
            push!(found, ReportDiagnostic(:sites_not_converged, :warning, variables_display(keys), string("the natural-gradient steps of ", failed, " of ", total, " sites had not settled at the last sweep: the sites are not yet at their targets. More `iterations`, or a smaller `alpha`, may help")))
            continue
        end
        projection = method === :moment_matching ? "moment matching" : "Laplace"
        push!(found, ReportDiagnostic(:sites_not_converged, :warning, variables_display(keys), string(failed, " of ", total, " site projections (", projection, ") did not converge. ", uppercasefirst(hint), " may help")))
    end
    for (name, keys) in unprojected
        n = unprojected_count[name]
        push!(found, ReportDiagnostic(:unprojected_messages, :note, variables_display(keys), string(n, n == 1 ? " message" : " messages", " had no mode and stayed log-densities, pulled back through the next node")))
    end
    return found
end

# How to give a projection more iterations: through the scheme the inference was given
maxiters_hint(::Nothing) = "a larger `maxiters` of the projection"
maxiters_hint(scheme) = string("`", nameof(typeof(scheme)), "(maxiters = …)`")

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
