# [What was approximated: the inference report](@id user-guide-inference-report)

An inference that mixes exact rules with approximations, such as [`LaplaceApproximation`](@ref)
for the nodes without rules, raises a question its result alone does not answer: which parts of
the posterior are exact, and which were approximated, and how? `infer(...; report = true)` records
the answer as an [`InferenceReport`](@ref), read as `result.report`, or as
`engine.model.metadata[:report]` for streaming inference. It has two sections:

- `result.report.approximations`, the [`ApproximationsReport`](@ref). It lists each approximated
  posterior, how it was projected and which nodes' messages formed it, exact or by the rule
  fallback. It also lists the joints and energies the fallback computed, and the warnings of
  [`RxInfer.diagnostics`](@ref).
- `result.report.timings`, the [`TimingsReport`](@ref): where the time went.

The report is collected from the engine's events, which are not built when it is off, the default.
`report = (:approximations,)` names the sections to record.

## A model with an approximation

The classifier below observes Bernoulli outcomes through the logistic function, a node with no
rules, so `fallback = LaplaceApproximation()` approximates it:

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
result = infer(model = classifier(), data = (y = y,), fallback = LaplaceApproximation(), report = true, free_energy = true)
result.report
```

How to read it:
- **The summary line.** One posterior, `x`'s, was approximated, and nothing calls for a warning.
- **Approximated posteriors.** Each approximated posterior is a group of rows. Its header says how
  it was projected: here Laplace's method, how many Newton iterations it took and the
  log-normaliser it found, which is the posterior's log evidence. Each row is a node instance as
  the model writes it, `y[1:8] ~ LogisticBernoulli(x)`, with instances of one shape folded.
  - The columns say how many of its messages towards the variable came from an exact rule and how
    many from the rule fallback, and where the fallback's expectations went.
  - Here every message from the likelihood is exact: the other input is an observation, so the
    expectation is an exact sum.
- **Joints and energies by the fallback.** The average energies that the free energy needed, also
  computed by the fallback.
- **Timings.** How the inference's time divides among the exact rules, the fallback, the
  projections, the free energy and the engine itself. Each call's own time is counted, so the
  kinds add up. The first run of a model also includes Julia's compilation.

The display prints the tables as aligned text. With PrettyTables.jl loaded, it prints them with
borders, the warnings highlighted:

```@example inference-report
using PrettyTables
result.report.approximations
```

## A conjugate model has nothing approximated

A rule that exists always takes precedence over the fallback, and a projection passes a closed
form as it is. So a conjugate model under the same keyword reports no approximated posterior:

```@example inference-report
@model function coin(y)
    p ~ Beta(1.0, 1.0)
    y .~ Bernoulli(p)
end

infer(model = coin(), data = (y = y,), fallback = LaplaceApproximation(), report = (:approximations,)).report
```

## Diagnostics

[`RxInfer.diagnostics`](@ref) lists the parts of an inference most likely to be inaccurate, as
warnings, and others worth knowing, as notes. Each is a [`ReportDiagnostic`](@ref) with its kind,
its level, the variable or node it is about, and a sentence. The display shows them first. Here
Laplace's method is allowed two iterations only, so it stops before it converges:

```@example inference-report
stopped = infer(model = classifier(), data = (y = y,), fallback = LaplaceApproximation(maxiters = 2), report = (:approximations,))
RxInfer.diagnostics(stopped.report)
```

```@example inference-report
stopped.report
```

The warnings:
- a posterior's Laplace approximation that did not converge, `:laplace_not_converged`;
- a joint computed by the fallback that did not converge, `:joint_not_converged`;
- under belief propagation, a fallback's message that is an expectation over another random
  variable's message, `:expectation_over_messages`. Such a message is the variational one, an
  approximation of the belief propagation message. Cavity methods, such as expectation
  propagation, are exact there.

The notes:
- messages left unprojected because Laplace found no mode for them, `:unprojected_messages`,
  which the fallback then pulled back through the next node;
- expectations by a stochastic backend, `:stochastic_expectation`;
- expectations by a backend registered as a last resort, `:last_resort_expectation`.

A test, or a notebook, can assert that an inference has no warnings:

```julia
@assert !any(d -> d.level === :warning, RxInfer.diagnostics(result.report))
```

## The report as data

Everything the display shows is plain data:
- [`report_tables`](@ref) gives each section's tables as [`ReportTable`](@ref)s of strings, with
  their row groups and the level of each row;
- the records underneath give the counts: `result.report.approximations.instances`,
  `.variables` and `RxInfer.node_types(report)`, by node type.

For example, the number of times the fallback computed a message towards `x`:

```@example inference-report
RxInfer.node_types(result.report.approximations)[(LogisticBernoulli, :message, :x)].fallbacks
```

and the rows of the posteriors' table:

```@example inference-report
first(report_tables(result.report.approximations)).rows
```

For statistics over many runs, rather than one inference's timings, use the
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
RxInfer.render_report_table
RxInfer.NodeInstance
RxInfer.NodeRecord
RxInfer.VariableRecord
RxInfer.ProjectionRecord
RxInfer.node_types
RxInfer.form_outcome
RxInfer.RxInferReportCallbacks
```
