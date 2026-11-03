# `infer(...; fallback = scheme)`: one keyword sets the rule fallback, the default form constraint,
# the expectation backend and the initialization strategy, what the options give explicitly winning.

@testmodule FallbackSchemes begin
    using RxInfer, BayesBase, Distributions, StatsFuns
    const EA = Base.require(Base.PkgId(Base.UUID("9cdca625-6a81-42ba-9f59-d0fb182fc17f"), "ExpectationApproximations"))

    # A Bernoulli observation through the logistic function, a node with no rules
    struct LogisticBernoulli{X}
        x::X
    end
    BayesBase.logpdf(d::LogisticBernoulli, y) = logpdf(Bernoulli(logistic(d.x)), y)
    @define_factor_node(node = LogisticBernoulli, type = Stochastic, interfaces = [:out, :x])

    @model function classifier(y)
        x ~ NormalMeanVariance(0.0, 4.0)
        y .~ LogisticBernoulli(x)
    end

    @model function online_classifier(m, v, y)
        x ~ NormalMeanVariance(m, v)
        y ~ LogisticBernoulli(x)
    end

    export EA, LogisticBernoulli, classifier, online_classifier
end

@testitem "fallback: the options a scheme sets, and the ones it leaves" setup = [FallbackSchemes] begin
    using .FallbackSchemes
    import RxInfer: apply_fallback, ReactiveMPInferenceOptions, getrulefallback, getdefault_form_constraint, getcontext
    import DeltaMessagePassingRules: ExpectationRuleFallback, LaplaceForm

    scheme = LaplaceApproximation(tol = 1.0e-8, maxiters = 50)
    options = apply_fallback(convert(ReactiveMPInferenceOptions, nothing), scheme)
    @test getrulefallback(options) === ExpectationRuleFallback()
    @test getdefault_form_constraint(options) === LaplaceForm(1.0e-8, 50)
    # the session's backends, frozen into a type-stable router that chooses as EA's default does
    backend = getcontext(options).expectation
    @test backend isa BayesBase.ExpectationFirstOf
    @test isconcretetype(typeof(backend))
    for q in (Normal(0.0, 1.0), Gamma(2.0, 1.0), Beta(2.0, 3.0), Categorical([0.5, 0.5]))
        @test BayesBase.which_expectation_backend(backend, sin, q) == BayesBase.which_expectation_backend(EA.EADefault(), sin, q)
    end

    # what the options give explicitly wins
    pinned = EA.EADefault(8)
    given = convert(
        ReactiveMPInferenceOptions,
        (rulefallback = NodeFunctionRuleFallback(), default_form_constraint = PointMassFormConstraint(), context = (expectation = pinned,)),
    )
    options = apply_fallback(given, scheme)
    @test getrulefallback(options) === NodeFunctionRuleFallback()
    @test getdefault_form_constraint(options) isa PointMassFormConstraint
    @test getcontext(options).expectation === pinned
    # a pinned backend, and another service of the context kept
    options = apply_fallback(convert(ReactiveMPInferenceOptions, (context = (matrix_correction = 1.0,),)), LaplaceApproximation(expectation = pinned))
    @test getcontext(options) === (matrix_correction = 1.0, expectation = pinned)
    @test apply_fallback(given, nothing) === given
    @test repr(scheme) == "LaplaceApproximation(tol = 1.0e-8, maxiters = 50)"
end

@testitem "fallback: the initialization a scheme gives" setup = [FallbackSchemes] begin
    using .FallbackSchemes
    import RxInfer: fallback_initialization

    block = @initialization begin
        q(x) = NormalMeanVariance(1.0, 1.0)
    end
    @test fallback_initialization(nothing, LaplaceApproximation()) === FromPriors()
    @test fallback_initialization(block, LaplaceApproximation()) == FromPriors(block)
    # a strategy given wins, and a scheme without one keeps the block
    @test fallback_initialization(FromPriors(block), LaplaceApproximation(initialization = nothing)) == FromPriors(block)
    @test fallback_initialization(block, LaplaceApproximation(initialization = nothing)) === block
    @test fallback_initialization(nothing, LaplaceApproximation(initialization = nothing)) === nothing
    @test fallback_initialization(block, nothing) === block
end

@testitem "fallback: through infer, every rule context static" setup = [FallbackSchemes] begin
    using .FallbackSchemes
    import DeltaMessagePassingRules: ExpectationRuleFallback

    y = Float64[1, 1, 0, 1, 1, 0, 1, 1]
    contexts = Any[]
    record(event) = (push!(contexts, event.mapping.context); nothing)
    result = infer(model = classifier(), data = (y = y,), fallback = LaplaceApproximation(), callbacks = (after_message_rule_call = record,))
    reference = EA.laplace(x -> logpdf(Normal(0.0, 2.0), x) + sum(logpdf(LogisticBernoulli(x), yi) for yi in y), 0.0)
    @test mean(result.posteriors[:x]) ≈ mean(reference.distribution) rtol = 1.0e-8
    @test var(result.posteriors[:x]) ≈ var(reference.distribution) rtol = 1.0e-8
    @test !isempty(contexts)
    # the backend chosen once is in each node's context, whose type is concrete
    @test all(context -> isconcretetype(typeof(context)), contexts)
    @test all(context -> context.expectation isa BayesBase.ExpectationFirstOf, contexts)

    # not a scheme
    @test_throws "`ExpectationRuleFallback()` is a rule fallback" infer(model = classifier(), data = (y = y,), fallback = ExpectationRuleFallback())
    @test_throws "`fallback` takes a fallback scheme" infer(model = classifier(), data = (y = y,), fallback = 1)
end

@testitem "fallback: streaming inference" setup = [FallbackSchemes] begin
    using .FallbackSchemes

    y = Float64[1, 1, 0, 1, 1, 0, 1, 1]
    autoupdates = @autoupdates begin
        m, v = mean_var(q(x))
    end
    initialization = @initialization begin
        q(x) = NormalMeanVariance(0.0, 4.0)
    end
    function filtered(; kwargs...)
        engine = infer(;
            model = online_classifier(), datastream = from(y) |> map(NamedTuple{(:y,), Tuple{Float64}}, (yi) -> (y = yi,)),
            autoupdates = autoupdates, initialization = initialization, keephistory = length(y), historyvars = (x = KeepLast(),), kwargs...,
        )
        return engine.history[:x]
    end
    switched = filtered(fallback = LaplaceApproximation())
    by_options = filtered(
        options = (
            rulefallback = DeltaMessagePassingRules.ExpectationRuleFallback(), default_form_constraint = DeltaMessagePassingRules.LaplaceForm(),
            context = (expectation = EA.EADefault(),),
        ),
    )
    @test length(switched) == length(y)
    @test mean.(switched) == mean.(by_options)
    @test var.(switched) == var.(by_options)
end
