# `infer(...; report = true)`: the inference report. Its approximations section says which nodes
# ran exact rules and which the fallback, how each posterior was formed and projected, and what to
# worry about; its timings section where the time went. Both are tables, plain or pretty.

@testmodule InferenceReports begin
    using RxInfer, BayesBase, Distributions, StatsFuns, ExponentialFamily

    struct LogisticBernoulli{X}
        x::X
    end
    BayesBase.logpdf(d::LogisticBernoulli, y) = logpdf(Bernoulli(logistic(d.x)), y)
    @define_factor_node(node = LogisticBernoulli, type = Stochastic, interfaces = [:out, :x])

    @model function classifier(y)
        x ~ NormalMeanVariance(0.0, 4.0)
        y .~ LogisticBernoulli(x)
    end

    @model function coin(y)
        p ~ Beta(1.0, 1.0)
        y .~ Bernoulli(p)
    end

    # `y ~ Product(a, b)`, `y` Gaussian around `a b`: a node with two random inputs and no rules
    struct Product{A, B} <: ContinuousUnivariateDistribution
        a::A
        b::B
    end
    BayesBase.logpdf(d::Product, y) = -(y - d.a * d.b)^2 / 2
    BayesBase.insupport(::Product, y) = true
    @define_factor_node(node = Product, type = Stochastic, interfaces = [:out, :a, :b])

    @model function product(y)
        a ~ NormalMeanVariance(1.0, 1.0)
        b ~ NormalMeanVariance(1.0, 1.0)
        y ~ Product(a, b)
    end

    # A backend labelled stochastic that answers with points, for the diagnostic of one
    struct Noisy end
    const Answerable = Union{NormalMeanVariance, PointMass}
    BayesBase.supports_expectation(::Noisy, ::Type, ::Type{<:Answerable}) = true
    BayesBase.has_expectation_label(::Noisy, ::Type, ::Type{<:Answerable}, ::BayesBase.ExpectationStochastic) = true
    function BayesBase.prepare_expectation(::Noisy, q::NormalMeanVariance)
        s = sqrt(3 * var(q))
        return BayesBase.ExpectationPointSet((mean(q) - s, mean(q), mean(q) + s), (1 / 6, 2 / 3, 1 / 6))
    end
    BayesBase.prepare_expectation(::Noisy, q::PointMass) = BayesBase.ExpectationPointSet((mean(q),), (1.0,))

    shown(x) = sprint(show, MIME"text/plain"(), x)

    export LogisticBernoulli, classifier, coin, Product, product, Noisy, shown
end

@testitem "report: the nodes the fallback ran, and the projections" setup = [InferenceReports] begin
    using .InferenceReports
    y = Float64[1, 1, 0, 1, 1, 0, 1, 1]
    result = infer(model = classifier(), data = (y = y,), fallback = LocalLaplace(), report = true, logscales = true, free_energy = true)
    report = result.report
    @test report isa InferenceReport
    @test propertynames(report) == (:approximations, :timings)
    approximations = report.approximations
    @test approximations isa ApproximationsReport

    # every message towards x comes from the fallback, its data through the exact sum
    towards_x = RxInfer.node_types(approximations)[(LogisticBernoulli, :message, :x)]
    @test towards_x.rules == 0 && towards_x.fallbacks == length(y)
    @test towards_x.fallback isa DeltaMessagePassingRules.ExpectationRuleFallback
    @test only(values(towards_x.backends)).exact
    # the prior's node ran its rules
    prior = approximations.nodes[(NormalMeanVariance, :message, :out)]
    @test prior.rules == 1 && prior.fallbacks == 0
    @test prior.rule isa MessagePassingRulesBase.RuleSpec
    @test approximations.nodes[(LogisticBernoulli, :energy, nothing)].fallbacks == length(y)

    # x's marginal was projected by Laplace, whose normaliser is its log evidence
    x = approximations.variables[(:x, nothing)]
    @test x.constraint isa DeltaMessagePassingRules.LaplaceForm
    @test x.marginal.projections == 1
    @test x.marginal.outcome.method === :laplace && x.marginal.outcome.converged
    @test x.marginal.outcome.logZ ≈ getlogscale(result.posteriors[:x])
    @test approximations.expectation isa BayesBase.ExpectationFirstOf
    @test isempty(RxInfer.diagnostics(report))
end

@testitem "report: node instances and the provenance of a posterior" setup = [InferenceReports] begin
    using .InferenceReports
    y = Float64[1, 1, 0, 1, 1, 0, 1, 1]
    approximations = infer(model = classifier(), data = (y = y,), fallback = LocalLaplace(), report = (:approximations,)).report.approximations

    # one instance per node, labelled as the model writes it
    labels = sort!([instance.label for instance in values(approximations.instances)])
    @test labels == sort!(vcat(["x ~ NormalMeanVariance(0.0, 4.0)"], ["y[$i] ~ LogisticBernoulli(x)" for i in eachindex(y)]))
    likelihoods = [instance for instance in values(approximations.instances) if instance.node === LogisticBernoulli]
    @test RxInfer.instances_label(likelihoods) == "y[1:8] ~ LogisticBernoulli(x)"
    @test all(instance -> instance.records[(:message, :x)].fallbacks == 1, likelihoods)

    # x's posterior: the prior's message by its rule, the eight likelihoods' by the fallback
    x = approximations.variables[(:x, nothing)]
    @test sum(last, values(x.incoming)) == length(y) && sum(first, values(x.incoming)) == 1
    @test RxInfer.is_approximated(x)

    # the table of posteriors, a group per approximated posterior
    tables = report_tables(approximations)
    @test [table.title for table in tables] == ["Approximated posteriors"]
    posteriors = first(tables)
    outcome = x.marginal.outcome
    @test posteriors.groups == [1 => "x: Laplace, converged in $(outcome.iterations) iterations, log Z = $(round(outcome.logZ; sigdigits = 4))"]
    @test posteriors.rows == [["x ~ NormalMeanVariance(0.0, 4.0)", "out", "1", "0", ""], ["y[1:8] ~ LogisticBernoulli(x)", "x", "0", "8", "exact"]]
    @test RxInfer.report_summary(approximations) == "Approximations: 1 of 1 posteriors approximated, no warnings"
end

@testitem "report: deterministic nodes, anonymous variables and messages left unprojected" setup = [InferenceReports] begin
    using .InferenceReports, LinearAlgebra, StatsFuns
    @model function trials(y, doses)
        θ ~ MvNormalMeanCovariance(zeros(2), diagm([16.0, 400.0]))
        for j in eachindex(y)
            p[j] := logistic(dot([1.0, doses[j]], θ))
            y[j] ~ Bernoulli(p[j])
        end
    end
    doses, deaths = [-0.86, -0.3, -0.05, 0.73], [0.0, 1.0, 1.0, 1.0]
    approximations = infer(model = trials(doses = doses), data = (y = deaths,), fallback = LocalLaplace(), report = (:approximations,)).report.approximations

    labels = Set(RxInfer.instances_label([i]) for i in values(approximations.instances))
    @test "p[1] := logistic(·)" in labels && "· := dot([1.0, -0.86], θ)" in labels
    # the messages towards dot's output had no mode, so they stayed log-densities
    found = RxInfer.diagnostics(approximations)
    @test [d.kind for d in found] == [:unprojected_messages]
    @test only(found).level === :note && only(found).subject == "· (anonymous)"
    # θ's posterior: the prior by its rule, then dot through the fallback; dot's message on out is
    # evaluated, not integrated, so it is no expectation input
    rows = first(t for t in report_tables(approximations) if t.title == "Approximated posteriors").rows
    @test ["· := dot(…, θ)", "in2", "0", "4", "exact"] in rows
end

@testitem "report: a conjugate model has no approximation" setup = [InferenceReports] begin
    using .InferenceReports
    flips = Float64[1, 0, 0, 1, 0]
    result = infer(model = coin(), data = (y = flips,), report = (:approximations,))
    approximations = result.report.approximations
    @test all(record -> record.fallbacks == 0, values(approximations.nodes))
    @test approximations.nodes[(Bernoulli, :message, :p)].rules == length(flips)
    @test !RxInfer.is_approximated(approximations.variables[(:p, nothing)])
    @test approximations.expectation === nothing
    @test isempty(RxInfer.diagnostics(result.report))
    @test RxInfer.report_summary(approximations) == "Approximations: 0 of 1 posteriors approximated (exact: p), no warnings"
    @test isempty(report_tables(approximations))
    # under the switch too, the projection passes the closed form
    switched = infer(model = coin(), data = (y = flips,), report = true, fallback = LocalLaplace())
    @test all(record -> record.fallbacks == 0, values(switched.report.approximations.nodes))
    @test switched.report.approximations.variables[(:p, nothing)].marginal.outcome.method === :closed_form
end

@testitem "report: the diagnostics, each provoked" setup = [InferenceReports] begin
    using .InferenceReports
    y = Float64[1, 1, 0, 1, 1, 0, 1, 1]

    # Laplace stopped after two iterations
    found = RxInfer.diagnostics(infer(model = classifier(), data = (y = y,), fallback = LocalLaplace(maxiters = 2), report = true).report)
    @test [(d.kind, d.level, d.subject) for d in found] == [(:laplace_not_converged, :warning, "x")]

    # a fallback between two random variables under belief propagation, by a stochastic backend
    result = infer(model = product(), data = (y = 2.0,), fallback = LocalLaplace(expectation = Noisy()), report = true)
    found = RxInfer.diagnostics(result.report)
    @test [(d.kind, d.level) for d in found] == [(:expectation_over_messages, :warning), (:stochastic_expectation, :note)]
    @test all(d -> d.subject == "y ~ Product(a, b)", found)
    @test contains(first(found).text, "towards") && contains(last(found).text, "Noisy")
    tables = report_tables(result.report.approximations)
    @test first(tables).title == "Diagnostics" && first(tables).levels == [:warning, :note]

    # the free energy takes the joint of a and b by Laplace, which converged
    tables = report_tables(infer(model = product(), data = (y = 2.0,), fallback = LocalLaplace(), report = true, free_energy = true).report.approximations)
    joints = first(t for t in tables if t.title == "Joints and energies by the fallback")
    @test any(row -> row[2] == "joint of (a, b)" && row[4] == "joint Laplace, converged", joints.rows)

    # the same model under mean field takes expectations over marginals: no warning
    meanfield = infer(model = product(), data = (y = 2.0,), constraints = MeanField(), iterations = 5, fallback = LocalLaplace(), report = true)
    @test isempty(RxInfer.diagnostics(meanfield.report))
end

@testitem "report: the display, plain and with PrettyTables" setup = [InferenceReports] begin
    using .InferenceReports
    y = Float64[1, 1, 0, 1, 1, 0, 1, 1]
    result = infer(model = classifier(), data = (y = y,), fallback = LocalLaplace(maxiters = 2), report = true)
    shown_report = shown(result.report)
    @test startswith(shown_report, "Inference report\n  Approximations: 1 of 1 posteriors approximated, 1 warning")
    @test contains(shown_report, "Expectations: ") && contains(shown_report, "Timings: model ")
    # the plain rendering, whichever packages are loaded
    plain = join((sprint(io -> RxInfer.render_report_table_with(nothing, io, t)) for t in report_tables(result.report)), "\n")
    @test contains(plain, "! warning  x") && contains(plain, "y[1:8] ~ LogisticBernoulli(x)")
    @test !contains(plain, "╭")

    # the same tables through PrettyTables once it is loaded
    @eval using PrettyTables
    pretty = Base.invokelatest(shown, result.report)
    @test contains(pretty, "╭") && contains(pretty, "y[1:8] ~ LogisticBernoulli(x)") && contains(pretty, "Approximated posteriors")
    @test contains(Base.invokelatest(sprint, (io, r) -> PrettyTables.pretty_table(io, r), result.report), "Time by kind")
end

@testitem "report: timings" setup = [InferenceReports] begin
    using .InferenceReports
    y = Float64[1, 1, 0, 1, 1, 0, 1, 1]
    report = infer(model = classifier(), data = (y = y,), fallback = LocalLaplace(), iterations = 3, report = true).report
    timings = report.timings
    @test timings isa TimingsReport
    @test timings.model > 0 && timings.inference > 0 && length(timings.iterations) == 3
    # every rule call is timed once, as the approximations count it
    calls(f) = sum(f, (record for ((node, kind, target), record) in report.approximations.nodes if kind !== :energy); init = 0)
    @test timings.kinds[:fallback][2] == calls(r -> r.fallbacks) >= length(y)
    @test timings.kinds[:rules][2] == calls(r -> r.rules)
    @test timings.kinds[:projections][2] >= 3
    @test sum(first, values(timings.kinds)) <= timings.inference
    @test timings.nodes[(LogisticBernoulli, :fallback)][2] == timings.kinds[:fallback][2]
    @test isempty(timings.open)
    @test startswith(RxInfer.report_summary(timings), "Timings: model ") && contains(RxInfer.report_summary(timings), "(3 iterations)")
    @test [t.title for t in report_tables(timings)] == ["Time by kind", "Time by node and variable"]
end

@testitem "report: off by default, and its errors" setup = [InferenceReports] begin
    using .InferenceReports
    import ReactiveMP: listens
    flips = Float64[1, 0, 0, 1, 0]
    result = infer(model = coin(), data = (y = flips,))
    @test_throws "The inference report has not been recorded" result.report
    @test_throws "`:diagnostics` is not a section" infer(model = coin(), data = (y = flips,), report = (:diagnostics,))
    callbacks = RxInfer.RxInferReportCallbacks(InferenceReport())
    @test_throws "already contains a `:report` key" infer(model = coin(), data = (y = flips,), report = true, callbacks = callbacks)
    # the events a section does not record are never built for it
    approximations = RxInfer.RxInferReportCallbacks(InferenceReport((:approximations,)))
    @test !listens(approximations, RxInfer.BeforeIterationEvent) && !listens(approximations, ReactiveMP.BeforeMessageRuleCallEvent)
    @test listens(approximations, ReactiveMP.AfterMessageRuleCallEvent)
    @test listens(callbacks, RxInfer.BeforeIterationEvent) && listens(callbacks, ReactiveMP.BeforeMessageRuleCallEvent)
    @test_throws "as many cells as its header" ReportTable("t", ["a", "b"], [["1"]])
end

@testitem "report: streaming inference" setup = [InferenceReports] begin
    using .InferenceReports
    @model function online_coin(a, b, y)
        p ~ Beta(a, b)
        y ~ Bernoulli(p)
    end
    autoupdates = @autoupdates begin
        a, b = params(q(p))
    end
    engine = infer(
        model = online_coin(), datastream = from(Float64[1, 0, 1]) |> map(NamedTuple{(:y,), Tuple{Float64}}, (y) -> (y = y,)),
        autoupdates = autoupdates, initialization = @initialization(
            begin
                q(p) = Beta(1.0, 1.0)
            end
        ), report = true,
    )
    report = engine.model.metadata[:report]
    @test report.approximations.nodes[(Bernoulli, :message, :p)].rules == 3
    @test report.timings.kinds[:rules][2] >= 3
end
