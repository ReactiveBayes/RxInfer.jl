# Universal Laplace inference through `infer`: no rules and no methods for the nodes that
# lack them, only DeltaMessagePassingRules' `ExpectationRuleFallback`, its `LaplaceForm` for every
# random variable and an expectation backend in the context.

@testmodule UniversalLaplace begin
    using RxInfer
    # ExpectationApproximations comes with DeltaMessagePassingRules, not as a dependency of RxInfer
    const EA = Base.require(Base.PkgId(Base.UUID("9cdca625-6a81-42ba-9f59-d0fb182fc17f"), "ExpectationApproximations"))
    const ExpectationRuleFallback = DeltaMessagePassingRules.ExpectationRuleFallback
    const LaplaceForm = DeltaMessagePassingRules.LaplaceForm

    universal(; kwargs...) = (
        rulefallback = ExpectationRuleFallback(), default_form_constraint = LaplaceForm(),
        context = (expectation = EA.EADefault(),), kwargs...,
    )

    # Whether a fallback ran, from the rule each message, marginal and energy event names.
    function fallback_calls()
        calls = Ref(0)
        count(event) = (event.rule isa ExpectationRuleFallback && (calls[] += 1); nothing)
        return calls, (after_message_rule_call = count, after_marginal_rule_call = count, after_factor_bound_free_energy = count)
    end
end

@testitem "LaplaceForm: a form constraint with a warm start and a normaliser" setup = [UniversalLaplace] begin
    using BayesBase, ExponentialFamily
    import ReactiveMP: preprocess_form_constraints, constrain_form_logscale, default_form_check_strategy, default_prod_constraint, FormConstraintCheckLast
    U = UniversalLaplace

    @test default_form_check_strategy(U.LaplaceForm()) === FormConstraintCheckLast()
    @test default_prod_constraint(U.LaplaceForm()) === GenericProd()
    wrapped = preprocess_form_constraints(U.LaplaceForm())
    product = ProductOf(NormalMeanVariance(0.0, 1.0), ContinuousUnivariateLogPdf(x -> -(2.0 - x)^2))
    projected, logZ = constrain_form_logscale(wrapped, product)
    @test mean(projected) ≈ 4 / 3 && var(projected) ≈ 1 / 3
    @test logZ ≈ U.EA.laplace(x -> logpdf(product, x), 0.0).logZ
    # The context keeps the mode, where the next projection starts; a closed form changes nothing.
    @test wrapped.context.previous ≈ 4 / 3
    @test constrain_form_logscale(wrapped, NormalMeanVariance(1.0, 2.0)) == (NormalMeanVariance(1.0, 2.0), nothing)
    @test wrapped.context.previous ≈ 4 / 3
end

@testitem "universal Laplace: a linear-Gaussian chain equals the Kalman smoother" setup = [UniversalLaplace] begin
    using BayesBase, Distributions, ExponentialFamily, StableRNGs
    U = UniversalLaplace

    # The observation `y ~ Observed(x, v)`, a normal log-density with no rules: its messages come
    # from the fallback, exact for observed data, and each posterior from the projection.
    struct Observed{X, V}
        x::X
        v::V
    end
    BayesBase.logpdf(d::Observed, y) = logpdf(Normal(d.x, sqrt(d.v)), y)
    @define_factor_node(node = Observed, type = Stochastic, interfaces = [:out, :x, :v])

    @model function chain(y, observation)
        x[1] ~ NormalMeanVariance(0.0, 1.0)
        y[1] ~ observation(x[1], 0.3)
        for t in 2:length(y)
            x[t] ~ NormalMeanVariance(0.8 * x[t - 1], 0.5)
            y[t] ~ observation(x[t], 0.3)
        end
    end

    y = cumsum(randn(StableRNG(1), 30))
    exact = infer(model = chain(observation = NormalMeanVariance), data = (y = y,), free_energy = true)
    universal = infer(model = chain(observation = Observed), data = (y = y,), free_energy = true, options = U.universal())
    @test all(mean.(universal.posteriors[:x]) .≈ mean.(exact.posteriors[:x]))
    @test all(var.(universal.posteriors[:x]) .≈ var.(exact.posteriors[:x]))
    @test only(universal.free_energy) ≈ only(exact.free_energy)
    # One switch gives the same
    switched = infer(model = chain(observation = Observed), data = (y = y,), free_energy = true, fallback = LocalLaplace())
    @test mean.(switched.posteriors[:x]) == mean.(universal.posteriors[:x])
    @test switched.free_energy == universal.free_energy
end

@testitem "universal Laplace: logistic regression equals the Laplace fit of the posterior" setup = [UniversalLaplace] begin
    using BayesBase, Distributions, ExponentialFamily, StatsFuns
    U = UniversalLaplace

    # The bioassay of BDA3 §3.7: y[i] of n[i] animals die at the log dose x[i].
    struct BinomialLogistic{T, X, N}
        θ::T
        x::X
        n::N
    end
    BayesBase.logpdf(d::BinomialLogistic, y) = logpdf(Binomial(d.n, logistic(d.θ[1] + d.θ[2] * d.x)), y)
    @define_factor_node(node = BinomialLogistic, type = Stochastic, interfaces = [:out, :θ, :x, :n])

    x, n, y = [-0.86, -0.3, -0.05, 0.73], [5, 5, 5, 5], [0, 1, 3, 5]
    prior = MvNormalMeanCovariance([0.0, 0.0], [4.0 0.0; 0.0 100.0])
    @model function bioassay(y, x, n)
        θ ~ prior
        for i in eachindex(y)
            y[i] ~ BinomialLogistic(θ, x[i], n[i])
        end
    end

    result = infer(model = bioassay(x = x, n = n), data = (y = y,), options = U.universal(), logscales = true)
    logposterior(θ) = logpdf(prior, θ) + sum(logpdf(BinomialLogistic(θ, x[i], n[i]), y[i]) for i in eachindex(y))
    reference = U.EA.laplace(logposterior, [0.0, 0.0])
    posterior = result.posteriors[:θ]
    @test mean(posterior) ≈ mean(reference.distribution) rtol = 1.0e-8
    @test cov(posterior) ≈ cov(reference.distribution) rtol = 1.0e-8
    # Every likelihood message is exact, so the posterior's log scale is Laplace's evidence.
    @test getlogscale(posterior) ≈ reference.logZ rtol = 1.0e-8
    switched = infer(model = bioassay(x = x, n = n), data = (y = y,), fallback = LocalLaplace(), logscales = true)
    @test mean(switched.posteriors[:θ]) == mean(posterior)
    @test getlogscale(switched.posteriors[:θ]) == getlogscale(posterior)
end

@testitem "universal Laplace: the bioassay as Bernoulli trials, through `dot` and through a Delta node" setup = [UniversalLaplace] begin
    using BayesBase, Distributions, ExponentialFamily, LinearAlgebra, StatsFuns
    U = UniversalLaplace

    # The bioassay of BDA3 §3.7, its Binomials expanded into Bernoulli trials and written as it reads
    # on paper. The deterministic nodes, RxInfer's `dot` and a Delta node, go through the rule
    # fallback wherever their own rules do not apply. Each trial's likelihood is exact, so the
    # posterior must be Laplace's approximation of the joint, as with one fused likelihood node.
    x, n, y = [-0.86, -0.3, -0.05, 0.73], [5, 5, 5, 5], [0, 1, 3, 5]
    doses = reduce(vcat, [fill(x[i], n[i]) for i in eachindex(x)])
    outcomes = reduce(vcat, [[ones(y[i]); zeros(n[i] - y[i])] for i in eachindex(x)])
    prior = MvNormalMeanCovariance([0.0, 0.0], [16.0 0.0; 0.0 400.0])

    @model function bioassay_dot(y, doses)
        θ ~ prior
        for j in eachindex(y)
            p[j] := logistic(dot([1.0, doses[j]], θ))
            y[j] ~ Bernoulli(p[j])
        end
    end
    dose_response(θ, dose) = logistic(θ[1] + θ[2] * dose)
    @model function bioassay_delta(y, doses)
        θ ~ prior
        for j in eachindex(y)
            p[j] := dose_response(θ, doses[j])
            y[j] ~ Bernoulli(p[j])
        end
    end

    logposterior(θ) = logpdf(prior, θ) + sum(logpdf(Binomial(n[i], logistic(θ[1] + θ[2] * x[i])), y[i]) for i in eachindex(x))
    reference = U.EA.laplace(logposterior, [0.0, 0.0]).distribution
    for model in (bioassay_dot(doses = doses), bioassay_delta(doses = doses))
        posterior = infer(model = model, data = (y = outcomes,), fallback = LocalLaplace()).posteriors[:θ]
        @test mean(posterior) ≈ mean(reference) rtol = 1.0e-8
        @test cov(posterior) ≈ cov(reference) rtol = 1.0e-8
    end
end

@testitem "universal Laplace: a Delta node without an approximation method, every variable returned" setup = [UniversalLaplace] begin
    using BayesBase, Distributions, ExponentialFamily
    U = UniversalLaplace

    @model function warped(y)
        x ~ NormalMeanVariance(0.0, 1.0)
        z := tanh(x)
        y ~ NormalMeanVariance(z, 0.1)
    end

    # The message on z pulled back through tanh is exact, so the posterior of x is Laplace's of
    # the exact posterior. Towards z the fallback sends the moment-matched pushforward, a Gaussian.
    result = infer(model = warped(), data = (y = 0.5,), options = U.universal())
    reference = U.EA.laplace(x -> logpdf(Normal(0.0, 1.0), x) + logpdf(Normal(tanh(x), sqrt(0.1)), 0.5), 0.0)
    @test mean(result.posteriors[:x]) ≈ mean(reference.distribution) rtol = 1.0e-8
    @test var(result.posteriors[:x]) ≈ var(reference.distribution) rtol = 1.0e-8
    @test result.posteriors[:z] isa ExponentialFamily.UnivariateNormalDistributionsFamily
    switched = infer(model = warped(), data = (y = 0.5,), fallback = LocalLaplace())
    @test switched.posteriors[:x] == result.posteriors[:x]
    @test switched.posteriors[:z] == result.posteriors[:z]
end

@testitem "universal Laplace: a two-input Delta node has a free energy under belief propagation" setup = [UniversalLaplace] begin
    using BayesBase, Distributions, ExponentialFamily
    U = UniversalLaplace

    combine(a, b) = a * exp(b)
    @model function product_model(y)
        a ~ NormalMeanVariance(1.0, 1.0)
        b ~ NormalMeanVariance(0.0, 0.5)
        z := combine(a, b)
        y ~ NormalMeanVariance(z, 0.2)
    end
    result = infer(model = product_model(), data = (y = 1.5,), free_energy = true, iterations = 5, options = U.universal())
    @test all(isfinite, result.free_energy)
    @test last(result.posteriors[:a]) isa ExponentialFamily.UnivariateNormalDistributionsFamily
end

@testitem "universal Laplace: a rate times a known exposure, its free energy exact" setup = [UniversalLaplace] begin
    using BayesBase, Distributions, ExponentialFamily, SpecialFunctions

    # `y ~ Poisson(λ * t)` with `t` known: every message has a rule, but the joint of `*` over
    # (λ, t) has none. The fallback gives it as λ's marginal, its message times the node's own
    # rule towards it, and t's point mass, so the Bethe free energy of this tree is -log p(y).
    @model function exposure(y, t)
        λ ~ Gamma(shape = 2.0, rate = 1.0)
        y ~ Poisson(λ * t)
    end
    y, t = 7, 3.5
    result = infer(model = exposure(t = t), data = (y = y,), free_energy = true, fallback = LocalLaplace(), report = true)
    # the negative binomial evidence of a Gamma-Poisson count
    logevidence = loggamma(y + 2.0) - loggamma(2.0) - loggamma(y + 1.0) + 2.0 * log(1.0 / (1.0 + t)) + y * log(t / (1.0 + t))
    @test only(result.free_energy) ≈ -logevidence rtol = 1.0e-8
    @test mean(result.posteriors[:λ]) ≈ (2.0 + y) / (1.0 + t)
    tables = report_tables(result.report.approximations)
    joints = first(table for table in tables if table.title == "Joints and energies by the fallback")
    @test any(row -> row[4] == "closed form", joints.rows)
end

@testitem "universal Laplace: conjugate models never ask the fallback" setup = [UniversalLaplace] begin
    using BayesBase, Distributions, ExponentialFamily, StableRNGs
    U = UniversalLaplace

    @model function iid(y)
        μ ~ NormalMeanVariance(0.0, 100.0)
        τ ~ GammaShapeRate(1.0, 1.0)
        y .~ NormalMeanPrecision(μ, τ)
    end
    @model function coin(y)
        p ~ Beta(1.0, 1.0)
        y .~ Bernoulli(p)
    end
    constraints = @constraints begin
        q(μ, τ) = q(μ)q(τ)
    end
    initialization = @initialization begin
        q(τ) = GammaShapeRate(1.0, 1.0)
    end
    y = rand(StableRNG(2), Normal(1.0, 0.5), 20)
    flips = Float64.(rand(StableRNG(3), Bernoulli(0.3), 20))

    for (model, data, kwargs) in ((iid(), (y = y,), (; constraints, initialization, iterations = 10, free_energy = true)), (coin(), (y = flips,), (; free_energy = true)))
        calls, callbacks = U.fallback_calls()
        plain = infer(; model, data, kwargs...)
        universal = infer(; model, data, kwargs..., options = U.universal(), callbacks)
        @test calls[] == 0
        @test plain.posteriors == universal.posteriors
        @test plain.free_energy == universal.free_energy
        # One switch: the same, when the priors initialise both
        calls, callbacks = U.fallback_calls()
        from_priors = infer(; model, data, kwargs..., initialization = FromPriors(get(kwargs, :initialization, nothing)))
        switched = infer(; model, data, kwargs..., fallback = LocalLaplace(), callbacks)
        @test calls[] == 0
        @test switched.posteriors == from_priors.posteriors
        @test switched.free_energy == from_priors.free_energy
    end
end

@testitem "universal Laplace: constrained variables under mean field, against the domain experiment" setup = [UniversalLaplace] begin
    using BayesBase, Distributions, ExponentialFamily
    U = UniversalLaplace

    # Zero-inflated Poisson with natural priors, p ~ Beta(1, 1) and λ ~ Gamma(2, 1): the
    # projection finds the domains from the priors, and Laplace runs in logit and log
    # coordinates. A year of counts; the reference is the posterior on a grid. Mean field misses
    # the strong correlation of p and λ, so the gap is reported and guarded, not tight: these tests
    # check that the inference runs and is close; how close mean field can be is a separate question.
    struct ZeroInflatedPoisson{P, L}
        p::P
        λ::L
    end
    function BayesBase.logpdf(d::ZeroInflatedPoisson, y)
        y == 0 && return log(d.p + (1 - d.p) * exp(-d.λ))
        return log1p(-d.p) + logpdf(Poisson(d.λ), y)
    end
    @define_factor_node(node = ZeroInflatedPoisson, type = Stochastic, interfaces = [:out, :p, :λ])

    @model function drunk_monks(y)
        p ~ Beta(1.0, 1.0)
        λ ~ Gamma(2.0, 1.0)
        y .~ ZeroInflatedPoisson(p, λ)
    end
    counts = vcat(fill(0.0, 284), fill(1.0, 64), fill(2.0, 14), fill(3.0, 3))
    constraints = @constraints begin
        q(p, λ) = q(p)q(λ)
    end
    initialization = @initialization begin
        q(p) = Beta(1.0, 1.0)
        q(λ) = Gamma(2.0, 1.0)
    end
    result = infer(;
        model = drunk_monks(), data = (y = counts,), constraints, initialization, iterations = 20,
        free_energy = true, options = U.universal(),
    )
    qp, qλ = last(result.posteriors[:p]), last(result.posteriors[:λ])
    @test isfinite(last(result.free_energy))
    # Settled: the free energy changes by no more than rounding over the last iterations.
    @test abs(result.free_energy[end] - result.free_energy[end - 5]) < 1.0e-6 * abs(result.free_energy[end])

    histogram = Dict(y => count(==(y), counts) for y in unique(counts))
    ps, λs = range(0.001, 0.999; length = 400), range(0.002, 3.0; length = 400)
    logposterior(p, λ) = logpdf(Beta(1.0, 1.0), p) + logpdf(Gamma(2.0, 1.0), λ) + sum(n * logpdf(ZeroInflatedPoisson(p, λ), y) for (y, n) in histogram)
    weights = [exp(logposterior(p, λ)) for p in ps, λ in λs]
    weights ./= sum(weights)
    grid_p, grid_λ = sum(weights .* ps), sum(weights .* λs')
    sd_p, sd_λ = sqrt(sum(weights .* (ps .- grid_p) .^ 2)), sqrt(sum(weights .* (λs' .- grid_λ) .^ 2))
    @info "zero-inflated Poisson: universal Laplace (mean field) against the grid" mean(qp) grid_p std(qp) sd_p mean(qλ) grid_λ std(qλ) sd_λ
    # Today the means are within half a posterior standard deviation of the grid's.
    @test abs(mean(qp) - grid_p) < 0.5 * sd_p
    @test abs(mean(qλ) - grid_λ) < 0.5 * sd_λ
    # One switch and no initialization: the priors are the initial marginals written above
    switched = infer(; model = drunk_monks(), data = (y = counts,), constraints, iterations = 20, free_energy = true, fallback = LocalLaplace())
    @test last(switched.posteriors[:p]) == qp && last(switched.posteriors[:λ]) == qλ
    @test switched.free_energy == result.free_energy
end

@testitem "universal Laplace: Poisson regression equals the Laplace fit of the posterior" setup = [UniversalLaplace] begin
    using BayesBase, Distributions, ExponentialFamily, StableRNGs
    U = UniversalLaplace

    struct PoissonLogLinear{T, X}
        θ::T
        x::X
    end
    BayesBase.logpdf(d::PoissonLogLinear, y) = logpdf(Poisson(exp(d.θ[1] + d.θ[2] * d.x)), y)
    @define_factor_node(node = PoissonLogLinear, type = Stochastic, interfaces = [:out, :θ, :x])

    rng = StableRNG(4)
    x = collect(range(-1.0, 1.0; length = 40))
    y = [rand(rng, Poisson(exp(0.5 + 1.2 * xi))) for xi in x]
    prior = MvNormalMeanCovariance([0.0, 0.0], [10.0 0.0; 0.0 10.0])
    @model function counts(y, x)
        θ ~ prior
        for i in eachindex(y)
            y[i] ~ PoissonLogLinear(θ, x[i])
        end
    end
    result = infer(model = counts(x = x), data = (y = y,), options = U.universal())
    reference = U.EA.laplace(θ -> logpdf(prior, θ) + sum(logpdf(PoissonLogLinear(θ, x[i]), y[i]) for i in eachindex(y)), [0.0, 0.0])
    @test mean(result.posteriors[:θ]) ≈ mean(reference.distribution) rtol = 1.0e-8
    @test cov(result.posteriors[:θ]) ≈ cov(reference.distribution) rtol = 1.0e-8
    @test infer(model = counts(x = x), data = (y = y,), fallback = LocalLaplace()).posteriors[:θ] == result.posteriors[:θ]
end

@testitem "universal Laplace: stochastic volatility, against the global Laplace approximation" setup = [UniversalLaplace] begin
    using BayesBase, Distributions, ExponentialFamily, StableRNGs
    U = UniversalLaplace

    # y[t] ~ N(0, exp(x[t])) with an AR(1) log-volatility: the observation has no rules, the
    # transitions their exact ones. Projecting each message on its own is not the global Laplace
    # approximation, which `LaplacePropagation()` reaches (see propagation_tests.jl), so the gap is
    # reported and guarded.
    struct Volatility{X}
        x::X
    end
    BayesBase.logpdf(d::Volatility, y) = logpdf(Normal(0.0, exp(d.x / 2)), y)
    @define_factor_node(node = Volatility, type = Stochastic, interfaces = [:out, :x])

    rng = StableRNG(5)
    T, ϕ, s = 50, 0.9, 0.3
    xs = zeros(T)
    xs[1] = randn(rng)
    for t in 2:T
        xs[t] = ϕ * xs[t - 1] + sqrt(s) * randn(rng)
    end
    y = [exp(x / 2) * randn(rng) for x in xs]
    @model function volatility(y)
        x[1] ~ NormalMeanVariance(0.0, 1.0)
        y[1] ~ Volatility(x[1])
        for t in 2:length(y)
            x[t] ~ NormalMeanVariance(ϕ * x[t - 1], s)
            y[t] ~ Volatility(x[t])
        end
    end
    result = infer(model = volatility(), data = (y = y,), options = U.universal())
    function logposterior(x)
        value = logpdf(Normal(0.0, 1.0), x[1]) + logpdf(Volatility(x[1]), y[1])
        for t in 2:T
            value += logpdf(Normal(ϕ * x[t - 1], sqrt(s)), x[t]) + logpdf(Volatility(x[t]), y[t])
        end
        return value
    end
    reference = U.EA.laplace(logposterior, zeros(T))
    gap = sqrt(sum(abs2, mean.(result.posteriors[:x]) .- mean(reference.distribution)) / T)
    @info "stochastic volatility: RMSE of the means against the global Laplace" gap
    # about 0.19, with T = 50, ϕ = 0.9 and s = 0.3
    @test gap < 0.25
    @test infer(model = volatility(), data = (y = y,), fallback = LocalLaplace()).posteriors[:x] == result.posteriors[:x]
end
