# `infer(...; report = true)`: the inference report, whose approximations section says which nodes
# ran exact rules and which the fallback, and how each variable was projected.

@testmodule InferenceReports begin
    using RxInfer, BayesBase, Distributions, StatsFuns

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

    export LogisticBernoulli, classifier, coin
end

@testitem "report: the nodes the fallback ran, and the projections" setup = [InferenceReports] begin
    using .InferenceReports
    y = Float64[1, 1, 0, 1, 1, 0, 1, 1]
    result = infer(model = classifier(), data = (y = y,), fallback = LaplaceApproximation(), report = true, logscales = true, free_energy = true)
    report = result.report
    @test report isa InferenceReport
    @test propertynames(report) == (:approximations,)
    approximations = report.approximations
    @test approximations isa ApproximationsReport

    # every message towards x comes from the fallback, its data through the exact sum
    towards_x = approximations.nodes[(LogisticBernoulli, :message, :x)]
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

    shown = sprint(show, MIME"text/plain"(), report)
    @test contains(shown, "LogisticBernoulli, message towards :x: 0 / $(length(y))")
    @test contains(shown, "x: LaplaceForm; marginal method = laplace")
end

@testitem "report: a conjugate model has no approximation" setup = [InferenceReports] begin
    using .InferenceReports
    flips = Float64[1, 0, 0, 1, 0]
    result = infer(model = coin(), data = (y = flips,), report = (:approximations,))
    approximations = result.report.approximations
    @test all(record -> record.fallbacks == 0, values(approximations.nodes))
    @test approximations.nodes[(Bernoulli, :message, :p)].rules == length(flips)
    @test isempty(approximations.variables)
    @test approximations.expectation === nothing
    # under the switch too, the projection passes the closed form
    switched = infer(model = coin(), data = (y = flips,), report = true, fallback = LaplaceApproximation())
    @test all(record -> record.fallbacks == 0, values(switched.report.approximations.nodes))
    @test switched.report.approximations.variables[(:p, nothing)].marginal.outcome.method === :closed_form
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
    # the events it does not record are never built for it
    @test !listens(callbacks, RxInfer.BeforeIterationEvent)
    @test listens(callbacks, ReactiveMP.AfterMessageRuleCallEvent)
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
    @test engine.model.metadata[:report].approximations.nodes[(Bernoulli, :message, :p)].rules == 3
end
