# `infer(...; fallback = NaturalGradientPropagation())`: a site per edge, in the family of its
# cavity, moved by a damped natural-gradient step towards its target. Its fixed point is the
# variational Gaussian where a model has one latent variable, exact on a linear-Gaussian chain,
# and the best Gamma on a Gamma edge.

@testmodule NaturalGradientModels begin
    using RxInfer, BayesBase, Distributions, StatsFuns, SpecialFunctions, LinearAlgebra, StableRNGs
    const EA = Base.require(Base.PkgId(Base.UUID("9cdca625-6a81-42ba-9f59-d0fb182fc17f"), "ExpectationApproximations"))

    # The bioassay of Gelman et al., a binomial likelihood through the logistic, a node with no rules
    logbinomial(n, k) = loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)
    struct BinomialLogistic{T, X, N} <: DiscreteUnivariateDistribution
        θ::T
        x::X
        n::N
    end
    BayesBase.logpdf(d::BinomialLogistic, y) = (η = d.θ[1] + d.θ[2] * d.x; -y * log1pexp(-η) - (d.n - y) * log1pexp(η) + logbinomial(d.n, y))
    BayesBase.insupport(::BinomialLogistic, y) = true
    @define_factor_node(node = BinomialLogistic, type = Stochastic, interfaces = [:out, :θ, :x, :n])

    const doses, animals, deaths = [-0.86, -0.3, -0.05, 0.73], [5, 5, 5, 5], [0, 1, 3, 5]
    const prior_θ = MvNormalMeanCovariance(zeros(2), diagm([16.0, 400.0]))

    @model function bioassay(y, x, n)
        θ ~ prior_θ
        for i in eachindex(y)
            y[i] ~ BinomialLogistic(θ, x[i], n[i])
        end
    end

    logposterior(θ) = logpdf(prior_θ, θ) + sum(logpdf(BinomialLogistic(θ, doses[i], animals[i]), deaths[i]) for i in eachindex(deaths))

    # A Gaussian observation as a node of its own, so that it goes through the fallback
    struct GaussianObservation{X, R} <: ContinuousUnivariateDistribution
        x::X
        r::R
    end
    BayesBase.logpdf(d::GaussianObservation, y) = -(log2π + log(d.r) + (y - d.x)^2 / d.r) / 2
    BayesBase.insupport(::GaussianObservation, y) = true
    @define_factor_node(node = GaussianObservation, type = Stochastic, interfaces = [:out, :x, :r])

    @model function chain(y, observation)
        x[1] ~ NormalMeanVariance(0.0, 1.0)
        y[1] ~ observation(x[1], 0.5)
        for t in 2:length(y)
            x[t] ~ NormalMeanVariance(0.9 * x[t - 1], 0.3)
            y[t] ~ observation(x[t], 0.5)
        end
    end

    # A rate λ = exp(z) with a Poisson count: the count's message towards λ is a Gamma, the
    # pushforward of z's prior a log-normal, so λ's site is a Gamma
    @model function count_rate(y, m, v)
        z ~ NormalMeanVariance(m, v)
        λ := exp(z)
        y ~ Poisson(λ)
    end

    # Counts with a log-rate, through a node with no rules
    struct LogPoisson{X} <: DiscreteUnivariateDistribution
        x::X
    end
    BayesBase.logpdf(d::LogPoisson, k) = k * d.x - exp(d.x) - loggamma(k + 1)
    BayesBase.insupport(::LogPoisson, k) = true
    @define_factor_node(node = LogPoisson, type = Stochastic, interfaces = [:out, :x])

    @model function count_chain(y, φ, q)
        x[1] ~ NormalMeanVariance(0.5, 1.0)
        y[1] ~ LogPoisson(x[1])
        for t in 2:length(y)
            x[t] ~ NormalMeanVariance(φ * x[t - 1], q)
            y[t] ~ LogPoisson(x[t])
        end
    end

    # `y ~ N(x, 1/τ)` through a node with no rules: under mean field each message towards τ is the
    # expected log-density, `log τ / 2 - τ E[(y - x)²] / 2`, of the Gamma family
    struct PrecisionObservation{X, T} <: ContinuousUnivariateDistribution
        x::X
        τ::T
    end
    BayesBase.logpdf(d::PrecisionObservation, y) = (log(d.τ) - log2π - d.τ * (y - d.x)^2) / 2
    BayesBase.insupport(::PrecisionObservation, y) = true
    @define_factor_node(node = PrecisionObservation, type = Stochastic, interfaces = [:out, :x, :τ])

    @model function unknown_precision(y, observation)
        x ~ NormalMeanVariance(0.0, 10.0)
        τ ~ Gamma(shape = 2.0, rate = 1.0)
        for i in eachindex(y)
            y[i] ~ observation(x, τ)
        end
    end

    export EA, BinomialLogistic, bioassay, doses, animals, deaths, logposterior, GaussianObservation, chain, count_rate, LogPoisson, count_chain, PrecisionObservation, unknown_precision
end

@testitem "NaturalGradientPropagation: the bioassay's posterior is the variational Gaussian of the joint" setup = [NaturalGradientModels] begin
    using .NaturalGradientModels
    import DeltaMessagePassingRules: NaturalGradientForm

    # one latent variable: the fixed point is the Gaussian closest to the posterior, up to the
    # quadratures of the targets and of the reference; with 64 points per dimension in both
    reference = EA.gaussian_variational(logposterior, zeros(2); backend = EA.GaussHermite(64))
    policy = RxInfer.NaturalGradientSitePolicy(NaturalGradientForm(backend = EA.GaussHermite(64)))
    precise = infer(
        model = bioassay(x = doses, n = animals), data = (y = deaths,), iterations = 60, fallback = NaturalGradientPropagation(),
        options = (site_policy = policy,), free_energy = true,
    )
    q = last(precise.posteriors[:θ])
    @test mean(q) ≈ mean(reference.distribution) rtol = 1.0e-6
    @test cov(q) ≈ cov(reference.distribution) rtol = 1.0e-6
    # the free energy is Bethe's with average energies: here the negative evidence lower bound
    @test last(precise.free_energy) ≈ -reference.logZ rtol = 1.0e-6

    # with the scheme's own 16 points per dimension, the quadrature's error remains
    result = infer(model = bioassay(x = doses, n = animals), data = (y = deaths,), iterations = 60, fallback = NaturalGradientPropagation(), free_energy = true)
    @test mean(last(result.posteriors[:θ])) ≈ mean(reference.distribution) rtol = 1.0e-5
    @test last(result.free_energy) ≈ -reference.logZ rtol = 1.0e-6
    # below Laplace's, and the posterior of β skewed to the right: the variational mean is above the mode
    laplace = infer(model = bioassay(x = doses, n = animals), data = (y = deaths,), fallback = LocalLaplace(), free_energy = true)
    @test last(result.free_energy) < only(laplace.free_energy)
    @test mean(q)[2] > mean(laplace.posteriors[:θ])[2]
end

@testitem "NaturalGradientPropagation: exact on a linear-Gaussian chain, as the Kalman smoother" setup = [NaturalGradientModels] begin
    using .NaturalGradientModels
    y = [0.3, -0.2, 0.8, 1.1, 0.4, -0.5]
    through = infer(model = chain(observation = GaussianObservation), data = (y = y,), iterations = 40, fallback = NaturalGradientPropagation(), free_energy = true)
    exact = infer(model = chain(observation = NormalMeanVariance), data = (y = y,), free_energy = true)
    @test maximum(abs, mean.(last(through.posteriors[:x])) - mean.(exact.posteriors[:x])) < 1.0e-8
    @test maximum(abs, var.(last(through.posteriors[:x])) - var.(exact.posteriors[:x])) < 1.0e-8
    @test last(through.free_energy) ≈ only(exact.free_energy) rtol = 1.0e-8
end

@testitem "NaturalGradientPropagation: a Gamma site reaches the optimum of the free energy over Gammas" setup = [NaturalGradientModels] begin
    using .NaturalGradientModels
    using SpecialFunctions: digamma, trigamma, loggamma

    # λ's belief Gamma(a, b) is the count's message times its site from the log-normal: at the
    # fixed point a stationary point, here a minimum, of
    # F(a, b) = E_q[log q] - E_q[log Gamma(λ | y + 1, 1)] - E_q[log LogNormal(λ | m, v)]
    y, m, v = 4, 1.0, 0.5
    function free_energy(a, b)
        μ₁ = digamma(a) - log(b)
        E_log_q = -entropy(Gamma(a, 1 / b))
        E_log_lik = y * μ₁ - a / b - loggamma(y + 1.0)
        E_log_push = -μ₁ - (trigamma(a) + (μ₁ - m)^2) / (2v) - log(2π * v) / 2
        return E_log_q - E_log_lik - E_log_push
    end
    result = infer(model = count_rate(m = m, v = v), data = (y = y,), iterations = 60, fallback = NaturalGradientPropagation(), report = true)
    q = last(result.posteriors[:λ])
    @test q isa GammaShapeRate
    a, b = shape(q), rate(q)
    F = free_energy(a, b)
    for (da, db) in ((1.001, 1.0), (0.999, 1.0), (1.0, 1.001), (1.0, 0.999), (1.001, 1.001), (0.999, 0.999))
        @test free_energy(a * da, b * db) > F
    end
    # the gradient vanishes there, to the quadrature's precision
    h = 1.0e-5
    @test abs((free_energy(a + h, b) - free_energy(a - h, b)) / 2h) < 1.0e-7
    @test abs((free_energy(a, b + h) - free_energy(a, b - h)) / 2h) < 1.0e-7
    @test isempty(RxInfer.diagnostics(result.report))
end

@testitem "NaturalGradientPropagation: damping and halving keep every marginal proper on a long chain" setup = [NaturalGradientModels] begin
    using .NaturalGradientModels
    using StableRNGs
    import ReactiveMP

    φ, q, T = 0.95, 0.1, 1000
    rng = StableRNG(1)
    x = zeros(T); x[1] = rand(rng, Normal(0.5, 1.0))
    for t in 2:T
        x[t] = φ * x[t - 1] + rand(rng, Normal(0.0, sqrt(q)))
    end
    y = [rand(rng, Poisson(exp(v))) for v in x]
    # every site against its cavity when it is sent: the marginal the step keeps proper
    improper = Ref(0)
    function check(event)
        site, cavity = event.result, ReactiveMP.getdata(event.cavity)
        (site === nothing || ismissing(site) || ismissing(cavity)) && return nothing
        _, w = weightedmean_precision(prod(GenericProd(), site, cavity))
        w > 0 || (improper[] += 1)
        return nothing
    end
    result = infer(
        model = count_chain(φ = φ, q = q), data = (y = y,), iterations = 30, fallback = NaturalGradientPropagation(), free_energy = true,
        returnvars = (x = KeepLast(),), callbacks = (after_site_computation = check,), report = true,
    )
    @test improper[] == 0
    @test all(p -> var(p) > 0 && isfinite(mean(p)), result.posteriors[:x])
    @test all(isfinite, result.free_energy)
    @test abs(result.free_energy[end] - result.free_energy[end - 1]) < 1.0e-8 * abs(result.free_energy[end])
    @test isempty(RxInfer.diagnostics(result.report))
end

@testitem "NaturalGradientPropagation: mean field gives a positive variable a Gamma posterior" setup = [NaturalGradientModels] begin
    using .NaturalGradientModels
    using StableRNGs

    # The node has no rules, so every message towards τ is a raw expected log-density, whose site
    # is a Gamma: the fixed point is variational message passing's with the rules of
    # NormalMeanPrecision, whose messages are of the same families
    rng = StableRNG(3)
    y = 1.5 .+ randn(rng, 20) ./ 2
    run(observation, fallback) = infer(
        model = unknown_precision(observation = observation), data = (y = y,), constraints = MeanField(), iterations = 50,
        initialization = @initialization(
            begin
                q(x) = NormalMeanVariance(0.0, 10.0)
                q(τ) = GammaShapeRate(2.0, 1.0)
            end
        ), fallback = fallback, free_energy = true,
    )
    ngmp = run(PrecisionObservation, NaturalGradientPropagation())
    vmp = run(NormalMeanPrecision, nothing)
    τ = last(ngmp.posteriors[:τ])
    @test τ isa GammaShapeRate
    @test shape(τ) ≈ shape(last(vmp.posteriors[:τ])) rtol = 1.0e-8
    @test rate(τ) ≈ rate(last(vmp.posteriors[:τ])) rtol = 1.0e-8
    @test mean(last(ngmp.posteriors[:x])) ≈ mean(last(vmp.posteriors[:x])) rtol = 1.0e-8
    @test last(ngmp.free_energy) ≈ last(vmp.free_energy) rtol = 1.0e-8
end

@testitem "NaturalGradientPropagation: the options it sets, the report and streaming" setup = [NaturalGradientModels] begin
    using .NaturalGradientModels
    import RxInfer: apply_fallback, ReactiveMPInferenceOptions, getdefault_form_constraint, getcontext, getsite_policy, getrulefallback
    import DeltaMessagePassingRules: VariationalForm, LaplaceForm, NaturalGradientForm, ExpectationRuleFallback

    scheme = NaturalGradientPropagation(alpha = 0.3, beta = 0.1, max_step = 5.0, tol = 1.0e-6)
    options = apply_fallback(convert(ReactiveMPInferenceOptions, nothing), scheme)
    form = getdefault_form_constraint(options)
    @test form.q isa VariationalForm && form.q.nomode === :project && form.μ.nomode === :leave
    # the variational Gaussians take the inference's backend, the rules' context's
    @test form.q.backend === getcontext(options).expectation
    policy = getsite_policy(options)
    @test policy isa RxInfer.NaturalGradientSitePolicy
    @test (policy.form.alpha, policy.form.beta, policy.form.max_step, policy.form.tol) == (0.3, 0.1, 5.0, 1.0e-6)
    fallback = getrulefallback(options)
    @test fallback isa ExpectationRuleFallback && fallback.propagation isa NaturalGradientForm && fallback.energy === :average
    # the free energy is Bethe's, not expectation propagation's
    @test !RxInfer.cavity_free_energy(policy)
    @test RxInfer.cavity_free_energy(getsite_policy(apply_fallback(convert(ReactiveMPInferenceOptions, nothing), ExpectationPropagation())))
    @test !RxInfer.cavity_free_energy(nothing)
    # an explicit form wins
    given = apply_fallback(convert(ReactiveMPInferenceOptions, (default_form_constraint = LaplaceForm(),)), scheme)
    @test getdefault_form_constraint(given) === LaplaceForm()
    @test sprint(show, NaturalGradientPropagation()) == "NaturalGradientPropagation(alpha = 0.5, beta = 0.2, max_step = Inf, tol = 1.0e-8)"
    @test_throws ArgumentError NaturalGradientPropagation(beta = 1.0)
    @test_throws ArgumentError NaturalGradientPropagation(alpha = 0.0)

    # the report: the sites, their steps, and whether they have settled
    report = infer(model = bioassay(x = doses, n = animals), data = (y = deaths,), iterations = 60, fallback = NaturalGradientPropagation(), report = (:approximations,)).report
    θ = report.approximations.variables[(:θ, nothing)]
    @test θ.sites.outcome.method === :natural_gradient && θ.sites.projections == 4 * 60 && length(θ.settled) == 4
    posteriors(report) = only(t for t in report_tables(report.approximations) if t.title == "Approximated posteriors")
    @test posteriors(report).groups[1].second == "θ: natural-gradient propagation, 4 sites, 240 steps"
    @test isempty(RxInfer.diagnostics(report))
    # after a few sweeps the steps have not settled, which the report says
    early = infer(model = bioassay(x = doses, n = animals), data = (y = deaths,), iterations = 3, fallback = NaturalGradientPropagation(), report = (:approximations,)).report
    @test endswith(posteriors(early).groups[1].second, "4 not settled")
    found = only(RxInfer.diagnostics(early))
    @test found.kind === :sites_not_converged && contains(found.text, "4 of 4 sites had not settled")

    # streaming: each update's sweeps move x's site on its observation's edge
    @model function online(m, v, y)
        x ~ NormalMeanVariance(m, v)
        y ~ GaussianObservation(x, 0.5)
    end
    autoupdates = @autoupdates begin
        m, v = mean_var(q(x))
    end
    observations = [0.5, 0.7, 0.2]
    engine = infer(
        model = online(), datastream = from(observations) |> map(NamedTuple{(:y,), Tuple{Float64}}, (y) -> (y = y,)),
        autoupdates = autoupdates, initialization = @initialization(
            begin
                q(x) = NormalMeanVariance(0.0, 1.0)
            end
        ), fallback = NaturalGradientPropagation(), iterations = 40, keephistory = 3,
    )
    @test length(engine.history[:x]) == 3
    # the observation is Gaussian in x: the Kalman filter
    filtered = accumulate(observations; init = (0.0, 1.0)) do (m, v), y
        w = 1 / v + 1 / 0.5
        return ((m / v + y / 0.5) / w, 1 / w)
    end
    # each update keeps its iterations' posteriors
    posterior = last.(engine.history[:x])
    @test mean.(posterior) ≈ first.(filtered) rtol = 1.0e-8
    @test var.(posterior) ≈ last.(filtered) rtol = 1.0e-8
end
