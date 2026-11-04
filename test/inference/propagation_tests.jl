# Cavity propagation (U7a): `LaplacePropagation()` and `ExpectationPropagation()`, each random
# variable keeping one site per edge against its outbound message on the edge. Exact where the
# model is Gaussian, Laplace propagation's fixed point the global Laplace approximation, and
# expectation propagation against an exact grid, its free energy included.

@testmodule Propagation begin
    using RxInfer, BayesBase, Distributions, LinearAlgebra, StableRNGs
    using SpecialFunctions: loggamma
    using StatsFuns: log2π
    const EA = Base.require(Base.PkgId(Base.UUID("9cdca625-6a81-42ba-9f59-d0fb182fc17f"), "ExpectationApproximations"))

    # `y ~ GaussianObservation(x, r)`: a Gaussian likelihood known only by its log-density
    struct GaussianObservation{X, R} <: ContinuousUnivariateDistribution
        x::X
        r::R
    end
    BayesBase.logpdf(d::GaussianObservation, y) = -(log2π + log(d.r) + (y - d.x)^2 / d.r) / 2
    BayesBase.insupport(::GaussianObservation, y) = true
    @define_factor_node(node = GaussianObservation, type = Stochastic, interfaces = [:out, :x, :r])

    # stochastic volatility's observation, and counts with a log-rate
    struct SVObservation{X, M} <: ContinuousUnivariateDistribution
        x::X
        μ::M
    end
    BayesBase.logpdf(d::SVObservation, y) = (v = d.μ + d.x; -(log2π + v + y^2 * exp(-v)) / 2)
    BayesBase.insupport(::SVObservation, y) = true
    @define_factor_node(node = SVObservation, type = Stochastic, interfaces = [:out, :x, :μ])

    struct LogPoisson{X} <: DiscreteUnivariateDistribution
        x::X
    end
    BayesBase.logpdf(d::LogPoisson, k) = k * d.x - exp(d.x) - loggamma(k + 1)
    BayesBase.insupport(::LogPoisson, k) = true
    @define_factor_node(node = LogPoisson, type = Stochastic, interfaces = [:out, :x])

    @model function chain(y, φ, q, observation, parameter)
        x[1] ~ NormalMeanVariance(0.0, 1.0)
        y[1] ~ observation(x[1], parameter)
        for t in 2:length(y)
            x[t] ~ NormalMeanVariance(φ * x[t - 1], q)
            y[t] ~ observation(x[t], parameter)
        end
    end

    @model function count_chain(y, φ, q)
        x[1] ~ NormalMeanVariance(0.5, 1.0)
        y[1] ~ LogPoisson(x[1])
        for t in 2:length(y)
            x[t] ~ NormalMeanVariance(φ * x[t - 1], q)
            y[t] ~ LogPoisson(x[t])
        end
    end

    rms(a) = sqrt(sum(abs2, a) / length(a))
end

@testitem "propagation: a linear-Gaussian chain with raw observations is exact" setup = [Propagation] begin
    using BayesBase, Distributions, LinearAlgebra, StableRNGs
    P = Propagation

    T, φ, q, r = 25, 0.9, 0.3, 0.5
    rng = StableRNG(3)
    x = zeros(T); x[1] = randn(rng)
    for t in 2:T
        x[t] = φ * x[t - 1] + sqrt(q) * randn(rng)
    end
    y = x .+ sqrt(r) .* randn(rng, T)
    # the exact posterior and evidence from the dense joint
    Λ = zeros(T, T); Λ[1, 1] = 1.0
    for t in 2:T
        Λ[t, t] += 1 / q; Λ[t - 1, t - 1] += φ^2 / q; Λ[t, t - 1] -= φ / q; Λ[t - 1, t] -= φ / q
    end
    Σ = inv(Λ + I / r)
    m = Σ * (y ./ r)
    logevidence = logpdf(MvNormal(zeros(T), Symmetric(inv(Λ) + r * I)), y)
    for scheme in (LaplacePropagation(), ExpectationPropagation())
        result = infer(model = P.chain(φ = φ, q = q, observation = P.GaussianObservation, parameter = r), data = (y = y,), iterations = 3, fallback = scheme, free_energy = true, returnvars = (x = KeepLast(),))
        @test maximum(abs, mean.(result.posteriors[:x]) - m) < 1.0e-8
        @test maximum(abs, var.(result.posteriors[:x]) - diag(Σ)) < 1.0e-8
        @test last(result.free_energy) ≈ -logevidence rtol = 1.0e-8
    end
end

@testitem "propagation: Laplace propagation's fixed point is the global Laplace approximation" setup = [Propagation] begin
    using BayesBase, Distributions, LinearAlgebra, StableRNGs
    P = Propagation

    T, μ, φ, σ = 50, -1.0, 0.95, 0.3
    rng = StableRNG(1234)
    x = zeros(T); x[1] = rand(rng, Normal(0, σ / sqrt(1 - φ^2)))
    for t in 2:T
        x[t] = φ * x[t - 1] + σ * randn(rng)
    end
    y = [exp((μ + x[t]) / 2) * randn(rng) for t in 1:T]
    function logjoint(x)
        s = logpdf(Normal(0, σ / sqrt(1 - φ^2)), x[1])
        for t in 2:T
            s += logpdf(Normal(φ * x[t - 1], σ), x[t])
        end
        return s + sum(logpdf(P.SVObservation(x[t], μ), y[t]) for t in 1:T)
    end
    reference = P.EA.laplace(logjoint, zeros(T)).distribution
    @model function sv(y, μ, φ, σ)
        x[1] ~ NormalMeanVariance(0.0, σ^2 / (1 - φ^2))
        y[1] ~ P.SVObservation(x[1], μ)
        for t in 2:length(y)
            x[t] ~ NormalMeanVariance(φ * x[t - 1], σ^2)
            y[t] ~ P.SVObservation(x[t], μ)
        end
    end
    result = infer(model = sv(μ = μ, φ = φ, σ = σ), data = (y = y,), iterations = 10, fallback = LaplacePropagation(), returnvars = (x = KeepLast(),))
    @test maximum(abs, mean.(result.posteriors[:x]) - mean(reference)) < 1.0e-6
    @test maximum(abs, std.(result.posteriors[:x]) - sqrt.(diag(cov(reference)))) < 1.0e-6
end

@testitem "propagation: expectation propagation on a count series, against the exact grid" setup = [Propagation] begin
    using BayesBase, Distributions, StableRNGs
    P = Propagation

    φ, q, T = 0.9, 0.1, 60
    rng = StableRNG(42)
    x = zeros(T); x[1] = rand(rng, Normal(0.5, 1.0))
    for t in 2:T
        x[t] = φ * x[t - 1] + rand(rng, Normal(0.0, sqrt(q)))
    end
    y = [rand(rng, Poisson(exp(v))) for v in x]
    # the exact smoothing marginals and evidence, by forward-backward on a grid
    grid = range(-5, 5, length = 801)
    h = step(grid)
    K = [pdf(Normal(φ * u, sqrt(q)), v) * h for u in grid, v in grid]
    likelihood(t) = [exp(logpdf(P.LogPoisson(u), y[t])) for u in grid]
    α, c = zeros(length(grid), T), zeros(T)
    a = [pdf(Normal(0.5, 1.0), u) * h for u in grid] .* likelihood(1)
    c[1] = sum(a); α[:, 1] = a ./ c[1]
    for t in 2:T
        a = (K' * α[:, t - 1]) .* likelihood(t)
        c[t] = sum(a); α[:, t] = a ./ c[t]
    end
    β = ones(length(grid), T)
    for t in (T - 1):-1:1
        β[:, t] = K * (likelihood(t + 1) .* β[:, t + 1]) ./ c[t + 1]
    end
    Pt = α .* β
    Pt ./= sum(Pt; dims = 1)
    exact = [sum(Pt[:, t] .* grid) for t in 1:T]

    result = infer(model = P.count_chain(φ = φ, q = q), data = (y = y,), iterations = 10, fallback = ExpectationPropagation(), free_energy = true, returnvars = (x = KeepLast(),), report = true)
    @test P.rms(mean.(result.posteriors[:x]) - exact) < 1.0e-3
    @test abs(last(result.free_energy) + sum(log, c)) < 0.1
    # the report: the posteriors were formed by expectation propagation's sites
    records = result.report.approximations.variables
    @test all(RxInfer.is_approximated, (records[(:x, t)] for t in 1:T))
    @test all(record -> record.sites.projections > 0, (records[(:x, t)] for t in 1:T))
    @test occursin("expectation propagation", RxInfer.projection_display([records[(:x, 1)]]))
end

@testitem "propagation: a two-dimensional chain through Delta nodes is exact where they are linear" setup = [Propagation] begin
    using BayesBase, Distributions, LinearAlgebra, StableRNGs, ExponentialFamily

    A = [1.0 0.05; -0.49 1.0]
    Q = 0.5 * [0.05^3 / 3 0.05^2 / 2; 0.05^2 / 2 0.05]
    linear(x) = A * x
    angle(x) = x[1]
    T, r = 25, 0.1
    rng = StableRNG(7)
    xs = [[1.5, 0.0]]
    for t in 2:T
        push!(xs, rand(rng, MvNormal(linear(xs[end]), Q)))
    end
    y = [rand(rng, Normal(x[1], sqrt(r))) for x in xs]
    @model function through_deltas(y, Q, r)
        x[1] ~ MvNormalMeanCovariance([1.5, 0.0], [0.1 0.0; 0.0 0.1])
        s[1] := angle(x[1])
        y[1] ~ NormalMeanVariance(s[1], r)
        for t in 2:length(y)
            x[t] ~ MvNormalMeanCovariance(linear(x[t - 1]), Q)
            s[t] := angle(x[t])
            y[t] ~ NormalMeanVariance(s[t], r)
        end
    end
    @model function exact_chain(y, Q, r)
        x[1] ~ MvNormalMeanCovariance([1.5, 0.0], [0.1 0.0; 0.0 0.1])
        y[1] ~ NormalMeanVariance(dot([1.0, 0.0], x[1]), r)
        for t in 2:length(y)
            x[t] ~ MvNormalMeanCovariance(A * x[t - 1], Q)
            y[t] ~ NormalMeanVariance(dot([1.0, 0.0], x[t]), r)
        end
    end
    exact = infer(model = exact_chain(Q = Q, r = r), data = (y = y,), free_energy = true)
    for scheme in (LaplacePropagation(), ExpectationPropagation(points = 10))
        result = infer(model = through_deltas(Q = Q, r = r), data = (y = y,), iterations = 3, fallback = scheme, free_energy = true, returnvars = (x = KeepLast(),))
        @test maximum(norm.(mean.(result.posteriors[:x]) - mean.(exact.posteriors[:x]))) < 1.0e-8
        @test maximum(norm.(cov.(result.posteriors[:x]) - cov.(exact.posteriors[:x]))) < 1.0e-8
        @test last(result.free_energy) ≈ only(exact.free_energy) rtol = 1.0e-8
    end
end

@testitem "propagation: the schemes set the site policy, and given options win" setup = [Propagation] begin
    import DeltaMessagePassingRules
    options = RxInfer.apply_fallback(convert(RxInfer.ReactiveMPInferenceOptions, (;)), ExpectationPropagation())
    @test RxInfer.getsite_policy(options) isa RxInfer.SitePolicy{<:DeltaMessagePassingRules.MomentForm}
    @test RxInfer.getrulefallback(options).propagation isa DeltaMessagePassingRules.MomentForm
    options = RxInfer.apply_fallback(convert(RxInfer.ReactiveMPInferenceOptions, (;)), LaplacePropagation())
    @test RxInfer.getsite_policy(options) isa RxInfer.SitePolicy{<:DeltaMessagePassingRules.LaplaceForm}
    given = RxInfer.SitePolicy(DeltaMessagePassingRules.LaplaceForm(); tol = 1.0e-6)
    options = RxInfer.apply_fallback(convert(RxInfer.ReactiveMPInferenceOptions, (site_policy = given,)), ExpectationPropagation())
    @test RxInfer.getsite_policy(options) === given
    @test isnothing(RxInfer.getsite_policy(RxInfer.apply_fallback(convert(RxInfer.ReactiveMPInferenceOptions, (;)), LaplaceApproximation())))
end
