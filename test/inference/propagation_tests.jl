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

    # `y ~ LogPrecisionNormal(μ, z)`: a Gaussian with mean μ and precision exp(z), a node with two
    # random inputs and no rules
    struct LogPrecisionNormal{M, Z} <: ContinuousUnivariateDistribution
        μ::M
        z::Z
    end
    BayesBase.logpdf(d::LogPrecisionNormal, y::Real) = (d.z - log(2π) - exp(d.z) * (y - d.μ)^2) / 2
    BayesBase.insupport(::LogPrecisionNormal, y) = true
    @define_factor_node(node = LogPrecisionNormal, type = Stochastic, interfaces = [:out, :μ, :z])

    @model function unknown_mean_precision(y)
        μ ~ NormalMeanVariance(0.0, 4.0)
        z ~ NormalMeanVariance(0.0, 1.0)
        for i in eachindex(y)
            y[i] ~ LogPrecisionNormal(μ, z)
        end
    end

    # x with a heavy-tailed prior and two count observations, through nodes without rules: every
    # edge of x is raw
    struct WidePrior{S} <: ContinuousUnivariateDistribution
        s::S
    end
    BayesBase.logpdf(d::WidePrior, x::Real) = -log1p((x / d.s)^2)
    BayesBase.insupport(::WidePrior, x) = true
    @define_factor_node(node = WidePrior, type = Stochastic, interfaces = [:out, :s])

    @model function wide_prior_mean(y)
        x ~ WidePrior(0.5)
        for i in eachindex(y)
            y[i] ~ NormalMeanVariance(x, 1.0)
        end
    end

    @model function counts_only(y)
        x ~ WidePrior(3.0)
        y[1] ~ LogPoisson(x)
        y[2] ~ LogPoisson(x)
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

@testitem "propagation: the free energy in streaming inference is each event's evidence" setup = [Propagation] begin
    using BayesBase, Distributions, StableRNGs
    P = Propagation

    # one step of a filter observed through a raw likelihood: each site is exact, so each event's
    # final free energy is the Kalman filter's predictive evidence, -log p(y_t | y_1:t-1)
    @model function filter_step(y, μ0, v0, φ, q, r)
        xprev ~ NormalMeanVariance(μ0, v0)
        x ~ NormalMeanVariance(φ * xprev, q)
        y ~ P.GaussianObservation(x, r)
    end
    autoupdates = @autoupdates begin
        μ0, v0 = mean_var(q(x))
    end
    φ, q, r, T, iterations = 0.9, 0.3, 0.5, 20, 3
    rng = StableRNG(5)
    x = zeros(T); x[1] = randn(rng)
    for t in 2:T
        x[t] = φ * x[t - 1] + sqrt(q) * randn(rng)
    end
    y = x .+ sqrt(r) .* randn(rng, T)
    function kalman(y)
        m, v = 0.0, 1.0
        evidence = Float64[]
        for yt in y
            mp, vp = φ * m, φ^2 * v + q
            push!(evidence, -logpdf(Normal(mp, sqrt(vp + r)), yt))
            k = vp / (vp + r)
            m, v = mp + k * (yt - mp), (1 - k) * vp
        end
        return evidence, m
    end
    evidence, m = kalman(y)
    for scheme in (LaplacePropagation(), ExpectationPropagation())
        stream, reported = Float64[], Float64[]
        engine = infer(
            model = filter_step(φ = φ, q = q, r = r), datastream = from(y) |> map(NamedTuple{(:y,), Tuple{Float64}}, (d) -> (y = d,)),
            autoupdates = autoupdates, initialization = @initialization(q(x) = NormalMeanVariance(0.0, 1.0)), returnvars = (:x,),
            historyvars = (x = KeepLast(),), keephistory = T, iterations = iterations, free_energy = Float64, fallback = scheme,
            callbacks = (on_free_energy_update = (event) -> push!(reported, event.value),), autostart = false,
        )
        subscription = subscribe!(engine.free_energy, (value) -> push!(stream, value))
        RxInfer.start(engine)
        # one value per iteration of each event, on the stream, in the history and to the callbacks
        @test length(stream) == T * iterations
        @test engine.free_energy_raw_history ≈ stream
        @test reported ≈ stream
        @test maximum(abs, engine.free_energy_final_only_history - evidence) < 1.0e-10
        @test mean(last(engine.history[:x])) ≈ m atol = 1.0e-10
        unsubscribe!(subscription)
        RxInfer.stop(engine)
    end
end

@testitem "propagation: early stopping reads expectation propagation's free energy" setup = [Propagation] begin
    using BayesBase, Distributions, StableRNGs
    P = Propagation

    φ, q, T = 0.9, 0.1, 40
    rng = StableRNG(42)
    x = zeros(T); x[1] = rand(rng, Normal(0.5, 1.0))
    for t in 2:T
        x[t] = φ * x[t - 1] + rand(rng, Normal(0.0, sqrt(q)))
    end
    y = [rand(rng, Poisson(exp(v))) for v in x]
    full = infer(model = P.count_chain(φ = φ, q = q), data = (y = y,), iterations = 20, fallback = ExpectationPropagation(), free_energy = true)
    stopped = infer(model = P.count_chain(φ = φ, q = q), data = (y = y,), iterations = 20, fallback = ExpectationPropagation(), free_energy = true, callbacks = (after_iteration = StopEarlyIterationStrategy(1.0e-10),))
    @test length(stopped.free_energy) < 20
    @test stopped.free_energy ≈ full.free_energy[1:length(stopped.free_energy)]
    @test last(stopped.free_energy) ≈ last(full.free_energy) rtol = 1.0e-8
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
    @test isnothing(RxInfer.getsite_policy(RxInfer.apply_fallback(convert(RxInfer.ReactiveMPInferenceOptions, (;)), LocalLaplace())))
end

@testitem "propagation: EP's quadrature, Gauss–Hermite or a given backend" setup = [Propagation] begin
    using Distributions, StableRNGs
    import ExpectationApproximations as EA
    @test ExpectationPropagation(points = 7).form.backend == EA.GaussHermite(7)
    scheme = ExpectationPropagation(backend = EA.SphericalRadialCubature())
    @test RxInfer.fallback_site(scheme).form.backend isa EA.SphericalRadialCubature
    @test RxInfer.fallback_form(scheme).μ.backend isa EA.SphericalRadialCubature
    @test repr(ExpectationPropagation(points = 7)) == "ExpectationPropagation(backend = GaussHermite(7), tol = 1.0e-10, maxiters = 100)"
    # the count chain with three points per projection, near the twenty points'
    P = Propagation
    φ, q, T = 0.9, 0.1, 30
    rng = StableRNG(42)
    x = zeros(T); x[1] = rand(rng, Normal(0.5, 1.0))
    for t in 2:T
        x[t] = φ * x[t - 1] + rand(rng, Normal(0.0, sqrt(q)))
    end
    y = [rand(rng, Poisson(exp(v))) for v in x]
    run(scheme) = mean.(infer(model = P.count_chain(φ = φ, q = q), data = (y = y,), iterations = 10, fallback = scheme, returnvars = (x = KeepLast(),)).posteriors[:x])
    @test P.rms(run(ExpectationPropagation(backend = EA.GenUnscented())) - run(ExpectationPropagation())) < 0.01
end

@testitem "propagation: a node with two random inputs sends belief propagation's message" setup = [Propagation] begin
    using BayesBase, Distributions, StableRNGs
    P = Propagation

    # an unknown mean and log-precision from 20 observations: expectation propagation with a site
    # per factor and variable, against the exact posterior on a grid
    rng = StableRNG(5)
    y = 1.0 .+ randn(rng, 20) ./ sqrt(exp(0.7))
    μs, zs = range(-3, 5; length = 801), range(-4, 4; length = 801)
    logp = [logpdf(Normal(0, 2), μ) + logpdf(Normal(0, 1), z) + sum(yi -> logpdf(P.LogPrecisionNormal(μ, z), yi), y) for μ in μs, z in zs]
    top = maximum(logp)
    w = exp.(logp .- top)
    evidence = top + log(sum(w) * step(μs) * step(zs))
    w ./= sum(w)
    mμ, mz = sum(w .* μs), sum(w .* zs')
    sμ, sz = sqrt(sum(w .* (μs .- mμ) .^ 2)), sqrt(sum(w .* (zs' .- mz) .^ 2))

    # the loop through the factors needs a first message, as loopy belief propagation does
    result = infer(
        model = P.unknown_mean_precision(), data = (y = y,), iterations = 10, fallback = ExpectationPropagation(),
        initialization = @initialization(begin
            μ(z) = NormalMeanVariance(0.0, 1.0)
        end), free_energy = true, returnvars = (μ = KeepLast(), z = KeepLast()),
    )
    qμ, qz = result.posteriors[:μ], result.posteriors[:z]
    @test abs(mean(qμ) - mμ) < 0.01 && abs(mean(qz) - mz) < 0.01
    @test abs(std(qμ) / sμ - 1) < 0.02 && abs(std(qz) / sz - 1) < 0.02
    @test abs(last(result.free_energy) + evidence) < 0.05
    @test result.free_energy[end] ≈ result.free_energy[end - 1] atol = 1.0e-8
end

@testitem "propagation: an initial message is every edge's first cavity" setup = [Propagation] begin
    using BayesBase, Distributions, StableRNGs
    P = Propagation

    # x with only raw edges: each raw message waits for a cavity, which only another edge's site
    # gives, so nothing starts without μ(x)
    y = [3, 5]
    @test_throws Exception infer(model = P.counts_only(), data = (y = y,), iterations = 5, fallback = ExpectationPropagation())
    grid = range(-10, 10; length = 20_001)
    logp = [-log1p((u / 3)^2) + sum(k * u - exp(u) for k in y) for u in grid]
    p = exp.(logp .- maximum(logp))
    p ./= sum(p)
    exact = sum(p .* grid)
    posteriors = map((NormalMeanVariance(0.0, 1.0), NormalMeanVariance(3.0, 10.0))) do initial
        result = infer(
            model = P.counts_only(), data = (y = y,), iterations = 20, fallback = ExpectationPropagation(), returnvars = (x = KeepLast(),),
            initialization = @initialization(begin
                μ(x) = initial
            end),
        )
        result.posteriors[:x]
    end
    # expectation propagation's fixed point, whichever message seeds it
    @test mean(posteriors[1]) ≈ mean(posteriors[2]) atol = 1.0e-6
    @test var(posteriors[1]) ≈ var(posteriors[2]) rtol = 1.0e-6
    @test abs(mean(posteriors[1]) - exact) < 0.02

    # a model that starts without one ends at the same fixed point with a broad one, which only
    # slows the first sweeps
    φ, q, T = 0.9, 0.1, 30
    rng = StableRNG(42)
    x = zeros(T); x[1] = rand(rng, Normal(0.5, 1.0))
    for t in 2:T
        x[t] = φ * x[t - 1] + rand(rng, Normal(0.0, sqrt(q)))
    end
    counts = [rand(rng, Poisson(exp(v))) for v in x]
    plain = infer(model = P.count_chain(φ = φ, q = q), data = (y = counts,), iterations = 40, fallback = ExpectationPropagation(), returnvars = (x = KeepLast(),))
    seeded = infer(
        model = P.count_chain(φ = φ, q = q), data = (y = counts,), iterations = 40, fallback = ExpectationPropagation(), returnvars = (x = KeepLast(),),
        initialization = @initialization(begin
            μ(x) = NormalMeanVariance(0.0, 100.0)
        end),
    )
    @test maximum(abs, mean.(plain.posteriors[:x]) - mean.(seeded.posteriors[:x])) < 1.0e-6
end

@testitem "propagation: a prior without rules on constants is projected again each round" setup = [Propagation] begin
    using BayesBase, Distributions
    P = Propagation

    # x ~ WidePrior(0.5), with Gaussian observations: the prior's message, from a constant, arrives
    # once, before the observations give it a cavity, and its site is projected again each round.
    # One raw site against a Gaussian cavity: expectation propagation is exact, and Laplace
    # propagation is the global Laplace approximation, LocalLaplace()'s.
    y = [2.0, 2.5, 1.5, 3.0]
    grid = range(-10, 10; length = 20_001)
    logp = [-log1p((u / 0.5)^2) + sum(yi -> -(yi - u)^2 / 2, y) for u in grid]
    p = exp.(logp .- maximum(logp))
    p ./= sum(p)
    m = sum(p .* grid)
    run(scheme) = infer(model = P.wide_prior_mean(), data = (y = y,), iterations = 10, fallback = scheme, returnvars = (x = KeepLast(),)).posteriors[:x]
    ep = run(ExpectationPropagation())
    @test mean(ep) ≈ m atol = 1.0e-5
    @test var(ep) ≈ sum(p .* (grid .- m) .^ 2) rtol = 1.0e-4
    lp, laplace = run(LaplacePropagation()), run(LocalLaplace())
    @test mean(lp) ≈ mean(laplace) atol = 1.0e-8
    @test var(lp) ≈ var(laplace) rtol = 1.0e-6
end
