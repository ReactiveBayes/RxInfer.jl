# The engine's hooks for universal inference, through `infer`: a default form constraint for every
# random variable, the rule fallback for a free energy and for a Delta node without an
# approximation method, and damping through the stream postprocessors.

@testitem "default_form_constraint: an inference option" begin
    import RxInfer: ReactiveMPInferenceOptions, getdefault_form_constraint, setdefault_form_constraint

    options = convert(ReactiveMPInferenceOptions, (;))
    @test getdefault_form_constraint(options) === nothing
    constraint = ReactiveMP.UnspecifiedFormConstraint()
    options = convert(ReactiveMPInferenceOptions, (default_form_constraint = constraint,))
    @test getdefault_form_constraint(options) === constraint
    @test getdefault_form_constraint(setdefault_form_constraint(options, nothing)) === nothing

    # One constraint for the messages and the marginal, or a `NamedTuple` of either.
    import RxInfer: default_form_constraints
    unspecified = ReactiveMP.UnspecifiedFormConstraint()
    @test default_form_constraints(nothing) === (unspecified, unspecified)
    @test default_form_constraints(constraint) === (constraint, constraint)
    @test default_form_constraints((q = constraint,)) === (unspecified, constraint)
    @test default_form_constraints((μ = constraint,)) === (constraint, unspecified)
    @test_throws "Unknown key of the `default_form_constraint` option: r" default_form_constraints((r = constraint,))
end

@testitem "default_form_constraint: every random variable without a constraint of its own" begin
    using BayesBase, Distributions
    import ReactiveMP: AbstractFormConstraint, FormConstraintCheckLast

    # Constraints that replace the posterior, so that it shows which one applied.
    struct Replace{D} <: AbstractFormConstraint
        distribution::D
    end
    ReactiveMP.constrain_form(c::Replace, _) = c.distribution
    ReactiveMP.default_form_check_strategy(::Replace) = FormConstraintCheckLast()
    ReactiveMP.default_prod_constraint(::Replace) = GenericProd()

    @model function two_coins(y, z)
        θ ~ Beta(1, 1)
        φ ~ Beta(1, 1)
        y ~ Bernoulli(θ)
        z ~ Bernoulli(φ)
    end

    data = (y = 1.0, z = 0.0)
    exact = infer(model = two_coins(), data = data)
    @test exact.posteriors[:θ] == Beta(2.0, 1.0)

    # The default applies to every random variable, and a variable's own constraint replaces it.
    default = infer(model = two_coins(), data = data, options = (default_form_constraint = Replace(Beta(5.0, 5.0)),))
    @test default.posteriors[:θ] == Beta(5.0, 5.0) && default.posteriors[:φ] == Beta(5.0, 5.0)
    own = infer(
        model = two_coins(), data = data, options = (default_form_constraint = Replace(Beta(5.0, 5.0)),),
        constraints = @constraints(
            begin
                q(θ)::Replace(Beta(3.0, 3.0))
            end
        ),
    )
    @test own.posteriors[:θ] == Beta(3.0, 3.0) && own.posteriors[:φ] == Beta(5.0, 5.0)
end

@testitem "default_form_constraint: a projection for every posterior of a model without rules" begin
    using ExponentialFamilyProjection, BayesBase, Distributions, StableRNGs

    # A Beta prior and a Bernoulli likelihood the engine has no rules for.
    struct MyBeta{A, B} <: ContinuousUnivariateDistribution
        a::A
        b::B
    end
    BayesBase.logpdf(d::MyBeta, x) = logpdf(Beta(d.a, d.b), x)
    BayesBase.insupport(d::MyBeta, x::Real) = true
    struct MyBernoulli{P} <: DiscreteUnivariateDistribution
        p::P
    end
    BayesBase.logpdf(d::MyBernoulli, x) = logpdf(Bernoulli(d.p), x)
    BayesBase.insupport(d::MyBernoulli, x::Real) = true
    @define_factor_node(node = MyBeta, type = Stochastic, interfaces = [:out, :a, :b])
    @define_factor_node(node = MyBernoulli, type = Stochastic, interfaces = [:out, :p])

    @model function coin(y, prior, likelihood)
        p ~ prior
        y .~ likelihood(p)
    end

    y = rand(StableRNG(42), Bernoulli(0.7), 100)
    exact = infer(model = coin(prior = Beta(1, 1), likelihood = Bernoulli), data = (y = y,))
    projected = infer(
        model = coin(prior = MyBeta(1, 1), likelihood = MyBernoulli), data = (y = y,),
        options = (rulefallback = NodeFunctionRuleFallback(), default_form_constraint = (q = ProjectedTo(Beta),)),
    )
    @test projected.posteriors[:p] isa Beta
    @test mean(projected.posteriors[:p]) ≈ mean(exact.posteriors[:p]) rtol = 5.0e-2
end

@testitem "rulefallback: a free energy through the context's expectation backend" begin
    using BayesBase, Distributions, ExponentialFamily

    # `y ~ Toy(a)`, an unnormalised unit-variance normal log-density in `y` with mean `a`, with a
    # rule towards `a` and no average energy.
    struct Toy{A}
        a::A
    end
    BayesBase.logpdf(d::Toy, x) = -(x - d.a)^2 / 2
    @define_factor_node(node = Toy, type = Stochastic, interfaces = [:out, :a])
    @define_message_update_rule(node = Toy, target = :a, args = (q[:out]::PointMass,), body = (args) -> NormalMeanVariance(mean(args.q[:out]), 1.0))

    # A three-point rule, exact for polynomials of degree five under a normal.
    struct ThreePoints end
    const Gaussian = ExponentialFamily.UnivariateNormalDistributionsFamily
    BayesBase.supports_expectation(::ThreePoints, ::Type, ::Type{<:Union{Gaussian, PointMass}}) = true
    function BayesBase.prepare_expectation(::ThreePoints, q::Gaussian)
        s = sqrt(3 * var(q))
        return BayesBase.ExpectationPointSet((mean(q) - s, mean(q), mean(q) + s), (1 / 6, 2 / 3, 1 / 6))
    end
    BayesBase.prepare_expectation(::ThreePoints, q::PointMass) = BayesBase.ExpectationPointSet((mean(q),), (1.0,))

    @model function toy(y)
        a ~ NormalMeanVariance(0.0, 1.0)
        y ~ Toy(a)
    end

    # Without a backend the average energy is missing; with one, the free energy is
    # -log ∫ N(a; 0, 1) Toy(a)(y) da = -log N(y; 0, 2) - log √(2π).
    @test_throws ReactiveMP.RuleNotFoundError infer(model = toy(), data = (y = 0.8,), free_energy = true, options = (rulefallback = NodeFunctionRuleFallback(),))
    result = infer(
        model = toy(), data = (y = 0.8,), free_energy = true, iterations = 3,
        options = (rulefallback = NodeFunctionRuleFallback(), context = (expectation = ThreePoints(),)),
    )
    @test mean(last(result.posteriors[:a])) ≈ 0.4 && var(last(result.posteriors[:a])) ≈ 0.5
    @test result.free_energy ≈ fill(-logpdf(Normal(0.0, sqrt(2.0)), 0.8) - log(2π) / 2, 3)
end

@testitem "rulefallback: a Delta node without an approximation method" begin
    using ExponentialFamilyProjection, BayesBase, Distributions, ExponentialFamily

    @model function warped(y)
        x ~ NormalMeanVariance(0.0, 1.0)
        z := tanh(x)
        y ~ NormalMeanVariance(z, 0.1)
    end

    # The posterior of x is ∝ N(x; 0, 1) N(y; tanh(x), 0.1). The projection minimises KL(q ‖ p),
    # so the reference is the same projection of that log-density, not its moments.
    y = 0.5
    reference = project_to(ProjectedTo(NormalMeanVariance), (x) -> logpdf(Normal(0.0, 1.0), x) + logpdf(Normal(tanh(x), sqrt(0.1)), y))

    # The fallback pulls the message on z back through tanh, and the projection gives the posterior
    # its form; a Delta node with no method is otherwise an error. Only x is asked for: towards z
    # the fallback plugs in a point mass, and BayesBase has no product of a point mass with the
    # message from y.
    @test_throws ReactiveMP.RuleNotFoundError infer(model = warped(), data = (y = y,), returnvars = (x = KeepLast(),))
    result = infer(
        model = warped(), data = (y = y,), returnvars = (x = KeepLast(),),
        options = (rulefallback = NodeFunctionRuleFallback(), default_form_constraint = ProjectedTo(NormalMeanVariance)),
    )
    @test result.posteriors[:x] isa NormalMeanVariance
    @test mean(result.posteriors[:x]) ≈ mean(reference) rtol = 5.0e-2
    @test var(result.posteriors[:x]) ≈ var(reference) rtol = 5.0e-2
end

@testitem "stream_postprocessors: damped coordinate ascent reaches the same posterior" begin
    using BayesBase, Distributions, ExponentialFamily, StableRNGs

    @model function iid(y)
        μ ~ NormalMeanVariance(0.0, 100.0)
        τ ~ GammaShapeRate(1.0, 1.0)
        y .~ NormalMeanPrecision(μ, τ)
    end
    constraints = @constraints begin
        q(μ, τ) = q(μ)q(τ)
    end
    initialization = @initialization begin
        q(τ) = GammaShapeRate(1.0, 1.0)
    end
    y = rand(StableRNG(1), Normal(1.5, 0.5), 20)
    fit(postprocessor, iterations) = infer(;
        model = iid(), data = (y = y,), constraints, initialization, iterations,
        options = (stream_postprocessors = postprocessor,),
    )

    plain = fit(nothing, 100)
    damped = fit(ReactiveMP.DampingPostprocessor(0.5), 100)
    @test mean(last(damped.posteriors[:μ])) ≈ mean(last(plain.posteriors[:μ]))
    @test mean(last(damped.posteriors[:τ])) ≈ mean(last(plain.posteriors[:τ]))
    # Damped, it is further from the fixed point after two iterations.
    target = mean(last(plain.posteriors[:μ]))
    @test abs(mean(last(fit(ReactiveMP.DampingPostprocessor(0.5), 2).posteriors[:μ])) - target) >
        abs(mean(last(fit(nothing, 2).posteriors[:μ])) - target)
end
