# `infer(...; fallback = CVI())`: the rules of `LaplaceApproximation`, with every belief that has no
# closed form projected onto its variational Gaussian, the one minimising KL(q ‖ p), rather than
# Laplace's at the mode.

@testmodule CVIModels begin
    using RxInfer, BayesBase, Distributions, StatsFuns, SpecialFunctions, LinearAlgebra
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

    const doses, animals, deaths = [-0.86, -0.30, -0.05, 0.73], [5, 5, 5, 5], [0, 1, 3, 5]
    const prior_θ = MvNormalMeanCovariance(zeros(2), diagm([16.0, 400.0]))

    @model function bioassay(y, x, n)
        θ ~ prior_θ
        for i in eachindex(y)
            y[i] ~ BinomialLogistic(θ, x[i], n[i])
        end
    end

    logposterior(θ) = logpdf(prior_θ, θ) + sum(logpdf(BinomialLogistic(θ, doses[i], animals[i]), deaths[i]) for i in eachindex(deaths))

    # `y ~ Poisson(exp(a + b))`: two random inputs, no rules; under mean field each conditional is
    # not Gaussian, so Laplace and the variational Gaussian differ
    struct LogLinearCount{A, B} <: DiscreteUnivariateDistribution
        a::A
        b::B
    end
    BayesBase.logpdf(d::LogLinearCount, y) = y * (d.a + d.b) - exp(d.a + d.b) - loggamma(y + 1)
    BayesBase.insupport(::LogLinearCount, y) = true
    @define_factor_node(node = LogLinearCount, type = Stochastic, interfaces = [:out, :a, :b])

    @model function counts(y)
        a ~ NormalMeanVariance(0.0, 1.0)
        b ~ NormalMeanVariance(0.0, 1.0)
        y .~ LogLinearCount(a, b)
    end

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

    export EA, BinomialLogistic, bioassay, doses, animals, deaths, logposterior, LogLinearCount, counts, GaussianObservation, chain
end

@testitem "CVI: the bioassay's posterior is the variational Gaussian of the joint" setup = [CVIModels] begin
    using .CVIModels
    result = infer(model = bioassay(x = doses, n = animals), data = (y = deaths,), fallback = CVI(), free_energy = true)
    q = result.posteriors[:θ]
    reference = EA.gaussian_variational(logposterior, zeros(2))
    @test mean(q) ≈ mean(reference.distribution) rtol = 1.0e-8
    @test cov(q) ≈ cov(reference.distribution) rtol = 1.0e-8
    # one latent variable: the free energy is the negative bound, below Laplace's
    @test only(result.free_energy) ≈ -reference.logZ rtol = 1.0e-6
    laplace = infer(model = bioassay(x = doses, n = animals), data = (y = deaths,), fallback = LaplaceApproximation(), free_energy = true)
    @test only(result.free_energy) < only(laplace.free_energy)
    # the posterior of β is skewed to the right: the variational mean is above the mode
    @test mean(q)[2] > mean(laplace.posteriors[:θ])[2]
end

@testitem "CVI: mean field over two random inputs, the free energy below Laplace's" setup = [CVIModels] begin
    using .CVIModels
    y = [3, 0, 1, 4, 2, 0, 1]
    run(fallback) = infer(model = counts(), data = (y = y,), constraints = MeanField(), iterations = 30, fallback = fallback, free_energy = true)
    cvi, laplace = run(CVI()), run(LaplaceApproximation())
    @test last(cvi.free_energy) < last(laplace.free_energy)
    @test abs(cvi.free_energy[end] - cvi.free_energy[end - 1]) < 1.0e-6 * abs(cvi.free_energy[end])
    @test isempty(RxInfer.diagnostics(infer(model = counts(), data = (y = y,), constraints = MeanField(), iterations = 5, fallback = CVI(), report = true).report))
end

@testitem "CVI: exact on a linear-Gaussian chain, as the Kalman smoother" setup = [CVIModels] begin
    using .CVIModels
    y = [0.3, -0.2, 0.8, 1.1, 0.4, -0.5]
    through = infer(model = chain(observation = GaussianObservation), data = (y = y,), fallback = CVI(), free_energy = true)
    exact = infer(model = chain(observation = NormalMeanVariance), data = (y = y,), free_energy = true)
    @test maximum(abs, mean.(through.posteriors[:x]) - mean.(exact.posteriors[:x])) < 1.0e-8
    @test maximum(abs, var.(through.posteriors[:x]) - var.(exact.posteriors[:x])) < 1.0e-8
    @test only(through.free_energy) ≈ only(exact.free_energy) rtol = 1.0e-8
end

@testitem "CVI: the options it sets, the report and streaming" setup = [CVIModels] begin
    using .CVIModels
    import RxInfer: apply_fallback, ReactiveMPInferenceOptions, getdefault_form_constraint, getcontext
    import DeltaMessagePassingRules: VariationalForm, LaplaceForm

    options = apply_fallback(convert(ReactiveMPInferenceOptions, nothing), CVI(tol = 1.0e-8, maxiters = 50))
    form = getdefault_form_constraint(options)
    @test form.q isa VariationalForm && form.q.nomode === :project && form.μ.nomode === :leave
    # the forms take the inference's backend, the rules' context's
    @test form.q.backend === getcontext(options).expectation
    @test form.q.tol == 1.0e-8 && form.q.maxiters == 50
    # an explicit form and backend win
    backend = EA.EADefault()
    given = apply_fallback(convert(ReactiveMPInferenceOptions, (default_form_constraint = LaplaceForm(), context = (expectation = backend,))), CVI())
    @test getdefault_form_constraint(given) === LaplaceForm()
    @test getcontext(given).expectation === backend
    # a backend in the options reaches the forms too
    @test getdefault_form_constraint(apply_fallback(convert(ReactiveMPInferenceOptions, (context = (expectation = backend,),)), CVI())).q.backend === backend
    @test sprint(show, CVI()) == "CVI(tol = 1.0e-10, maxiters = 100)"

    report = infer(model = bioassay(x = doses, n = animals), data = (y = deaths,), fallback = CVI(), report = (:approximations,)).report
    θ = report.approximations.variables[(:θ, nothing)]
    @test θ.constraint isa VariationalForm && θ.marginal.outcome.method === :variational && θ.marginal.outcome.converged
    @test startswith(first(report_tables(report.approximations)).groups[1].second, "θ: CVI, converged in ")
    @test contains(first(report_tables(report.approximations)).groups[1].second, "ELBO = ")
    @test isempty(RxInfer.diagnostics(report))

    @model function online(m, v, y)
        x ~ NormalMeanVariance(m, v)
        y ~ GaussianObservation(x, 0.5)
    end
    autoupdates = @autoupdates begin
        m, v = mean_var(q(x))
    end
    engine = infer(
        model = online(), datastream = from([0.5, 0.7, 0.2]) |> map(NamedTuple{(:y,), Tuple{Float64}}, (y) -> (y = y,)),
        autoupdates = autoupdates, initialization = @initialization(
            begin
                q(x) = NormalMeanVariance(0.0, 1.0)
            end
        ), fallback = CVI(), keephistory = 3,
    )
    @test length(engine.history[:x]) == 3
    @test all(q -> q isa NormalMeanVariance, engine.history[:x])
end
