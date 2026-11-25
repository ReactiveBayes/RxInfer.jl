# [What was approximated: the inference report](@id user-guide-inference-report)

When an inference combines exact rules with approximations, for example
[`LocalLaplace`](@ref) for the nodes without rules, the result alone does not show which parts of
the posterior are exact and which were approximated. `infer(...; report = true)` records this in
an [`InferenceReport`](@ref). The report is available as `result.report`, or as
`engine.model.metadata[:report]` for streaming inference, and it has two sections:

- `result.report.approximations`, an [`ApproximationsReport`](@ref), lists each approximated
  posterior, how it was projected, and which nodes' messages formed it, either by an exact rule or
  by the rule fallback. It also lists the joint marginals and average energies that the fallback
  computed, and the warnings of [`RxInfer.diagnostics`](@ref).
- `result.report.timings`, a [`TimingsReport`](@ref), shows where the time went.

The report is collected from the events of the engine. It is off by default, and then these events
are not created. `report = (:approximations,)` selects the sections to record.

## A model with an approximation

The classifier below observes Bernoulli outcomes through the logistic function. This node has no
rules, so `fallback = LocalLaplace()` approximates its messages:

```@example inference-report
using RxInfer

# A Bernoulli observation through the logistic function, a node with no rules
struct LogisticBernoulli{X}
    x::X
end
BayesBase.logpdf(d::LogisticBernoulli, y) = logpdf(Bernoulli(1 / (1 + exp(-d.x))), y)
@define_factor_node(node = LogisticBernoulli, type = Stochastic, interfaces = [:out, :x])

@model function classifier(y)
    x ~ NormalMeanVariance(0.0, 4.0)
    y .~ LogisticBernoulli(x)
end

y = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
result = infer(model = classifier(), data = (y = y,), fallback = LocalLaplace(), report = true, free_energy = true)
result.report
```

The first line of the approximations section is a summary: one posterior, that of `x`, was
approximated, and there are no warnings. The summary also names the posteriors that are exact and,
apart from them, those that exact rules computed from approximated messages: in
`s1 → s2 → y`, where `y`'s likelihood has no rules, `s2` is approximated, and `s1`, whose messages
come from the Gaussian node's exact rule but from `s2`'s belief, is listed under "exact rules on
approximated inputs" ([`RxInfer.depends_on_approximation`](@ref)).

In the table "Approximated posteriors", each approximated posterior is a group of rows. The group
header says how the posterior was projected: here by Laplace's method, with the number of Newton
iterations and the log-normalizer it found, which is the Laplace approximation of the log
evidence. Each row is a node instance as the model writes it, `y[1:8] ~ LogisticBernoulli(x)`,
with instances of the same shape folded into one row. The columns count the node's messages
towards the variable that came from an exact rule and from the rule fallback, and name the
backends of the fallback's expectations. Here the expectations are exact: the other input of the
likelihood is an observation, so each expectation is an exact sum.

Under [`LaplacePropagation`](@ref) and [`ExpectationPropagation`](@ref), the group header names the
scheme, "Laplace propagation" or "expectation propagation", and gives the number of sites projected
on the variable's edges and how many of them did not converge. Under
[`NaturalGradientPropagation`](@ref), it reads "natural-gradient propagation" and gives the number
of sites, the number of steps they took, and how many had not settled at the last sweep.

The table "Joints and energies by the fallback" lists what else the fallback computed, here the
average energies needed for the free energy.

The timings section divides the time of the inference among the exact rules, the fallback, the
projections, the free energy and the engine itself. Each call is counted with its own time only,
without the calls it makes, so the kinds add up to the total. The first run of a model also
includes Julia's compilation time.

The tables are printed as aligned text. With PrettyTables.jl loaded, they are printed with borders
and with the warnings highlighted:

```@example inference-report
using PrettyTables
result.report.approximations
```

## A conjugate model has nothing approximated

Existing rules take precedence over the fallback, and the projections leave closed forms
unchanged. A conjugate model run with the same keyword therefore reports no approximated
posteriors:

```@example inference-report
@model function coin(y)
    p ~ Beta(1.0, 1.0)
    y .~ Bernoulli(p)
end

infer(model = coin(), data = (y = y,), fallback = LocalLaplace(), report = (:approximations,)).report
```

## Diagnostics

[`RxInfer.diagnostics`](@ref) returns warnings for the parts of an inference that are most likely
to be inaccurate, and notes for other things worth knowing. Each is a [`ReportDiagnostic`](@ref)
with a kind, a level, the variable or node it concerns, and a one-sentence description. The
display of the report shows them first. In the following example, Laplace's method is limited to
two iterations, so it stops before it converges:

```@example inference-report
stopped = infer(model = classifier(), data = (y = y,), fallback = LocalLaplace(maxiters = 2), report = (:approximations,))
RxInfer.diagnostics(stopped.report)
```

```@example inference-report
stopped.report
```

The warnings are:
- `:laplace_not_converged`: the Laplace approximation of a posterior did not converge;
- `:variational_not_converged`: the variational Gaussian of a posterior did not converge; under
  [`NaturalGradientPropagation`](@ref), this is the projection of a belief without sites;
- `:moment_matching_not_converged`: the moment matching of a posterior did not converge, under
  [`ExpectationPropagation`](@ref);
- `:sites_not_converged`: under [`LaplacePropagation`](@ref) or [`ExpectationPropagation`](@ref),
  the projections of some of a variable's sites did not converge; under
  [`NaturalGradientPropagation`](@ref), the last natural-gradient step of some sites had not
  settled, so these sites are not yet at their targets;
- `:joint_not_converged`: a joint computed by the fallback did not converge;
- `:expectation_over_messages`: under belief propagation, a message of the fallback is an
  expectation over the message of another random variable. Such a message is the variational
  message, which approximates the belief propagation message. Under the cavity schemes, a node
  whose other inputs carry Gaussian messages or point masses sends the belief propagation message
  and is not flagged, but a node with an input that carries another message, such as a Gamma, is.

The notes are:
- `:unprojected_messages`: Laplace's method found no mode for some messages, so they were left
  unprojected, and the fallback pulled them back through the next node;
- `:stochastic_expectation`: some expectations were computed by a stochastic backend;
- `:last_resort_expectation`: some expectations were computed by a backend registered as a last
  resort.

To check in a test or a notebook that an inference has no warnings:

```julia
@assert !any(d -> d.level === :warning, RxInfer.diagnostics(result.report))
```

## The report as data

Everything the display shows is also available as data. [`report_tables`](@ref) returns the
tables of each section as [`ReportTable`](@ref)s of strings, with their row groups and the level of
each row. The underlying records hold the counts: `result.report.approximations.instances`,
`result.report.approximations.variables`, and `RxInfer.node_types(report)` for the counts by node
type.

For example, the number of times the fallback computed a message towards `x`:

```@example inference-report
RxInfer.node_types(result.report.approximations)[(LogisticBernoulli, :message, :x)].fallbacks
```

and the rows of the table of approximated posteriors:

```@example inference-report
first(report_tables(result.report.approximations)).rows
```

For timing statistics over many runs, use the
[benchmark callbacks](@ref manual-inference-benchmark-callbacks).

## Reference

```@docs
InferenceReport
ApproximationsReport
TimingsReport
RxInfer.diagnostics
ReportDiagnostic
report_tables
ReportTable
RxInfer.report_summary
RxInfer.depends_on_approximation
RxInfer.render_report_table
RxInfer.NodeInstance
RxInfer.NodeRecord
RxInfer.VariableRecord
RxInfer.ProjectionRecord
RxInfer.node_types
RxInfer.form_outcome
RxInfer.RxInferReportCallbacks
```
