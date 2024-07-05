@testitem "Simple prior-likelihood model check" begin
    using ExponentialFamilyProjection, StableRNGs, Plots, BayesBase, Distributions

    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    @model function simple_model_1(y, prior, likelihood)
        p ~ prior
        y .~ likelihood(p)
    end

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

    @node MyBeta Stochastic [out, a, b]
    @node MyBernoulli Stochastic [out, p]

    @constraints function projection_constraints()
        q(p)::ProjectedTo(Beta)
    end

    rng = StableRNG(42)

    function run_experiment(y)
        analytical = infer(model = simple_model_1(prior = Beta(1, 1), likelihood = Bernoulli), data = (y = y,))
        projected = infer(
            model = simple_model_1(prior = MyBeta(1, 1), likelihood = MyBernoulli),
            data = (y = y,),
            constraints = projection_constraints(),
            options = (rulefallback = NodeFunctionRuleFallback(),)
        )
        return analytical, projected
    end

    for n in (10, 100, 1000), p in (0.25, 0.5, 0.75)
        y = rand(StableRNG(42), Bernoulli(p), n)

        analytical, projected = run_experiment(y)

        @test mean(analytical.posteriors[:p]) ≈ mean(projected.posteriors[:p]) rtol = 1e-1
    end

    @test_plot "projection" "simple-beta-bernoulli" begin
        y = rand(StableRNG(42), Bernoulli(0.68), 1000)
        analytical, projected = run_experiment(y)
        p = plot(0.0:0.01:1.0, (x) -> pdf(analytical.posteriors[:p], x), label = "analytical posterior", fill = 0, fillalpha = 0.2)
        p = plot!(p, 0.0:0.01:1.0, (x) -> pdf(projected.posteriors[:p], x), label = "projected posterior", fill = 0, fillalpha = 0.2)
        return p
    end
end

@testitem "Non-conjugate IID estimation" begin
    using StableRNGs, ExponentialFamilyProjection, BayesBase, Distributions, Plots

    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    @model function non_conjugate_iid(y)
        t ~ Beta(2, 2)
        r ~ Beta(3, 3)
        y .~ Normal(mean = t, prec = r)
    end

    @constraints function non_conjugate_iid_constraints()
        q(t)::ProjectedTo(Beta)
        q(r)::ProjectedTo(Beta)
        q(t, r) = q(t)q(r)
    end

    for realt in (0.15, 0.67), realr in (0.23, 0.90), n in (5000, 10000)
        ydata = rand(StableRNG(43), NormalMeanPrecision(realt, realr), n)

        result = infer(
            model = non_conjugate_iid(),
            data = (y = ydata,),
            constraints = non_conjugate_iid_constraints(),
            initialization = @initialization(q(t) = Beta(2, 2)),
            returnvars = KeepLast(),
            iterations = 5,
            free_energy = true,
            options = (limit_stack_depth = 500,)
        )

        @test mean(result.posteriors[:r]) ≈ realr atol = 1e-2
        @test mean(result.posteriors[:t]) ≈ realt atol = 1e-1
        @test result.free_energy[1] > result.free_energy[end]
        @test all(d -> d < 0 || abs(d) < 1e-3, diff(result.free_energy))
    end

    @test_plot "projection" "non_conjugate_iid" begin
        realr = 0.2
        realt = 0.9
        ydata = rand(StableRNG(43), NormalMeanPrecision(realt, realr), 1000)
        result = infer(
            model = non_conjugate_iid(),
            data = (y = ydata,),
            constraints = non_conjugate_iid_constraints(),
            initialization = @initialization(q(t) = Beta(2, 2)),
            returnvars = KeepLast(),
            iterations = 5,
            free_energy = true,
            options = (limit_stack_depth = 500,)
        )
        p1 = plot(0.0:0.01:1.0, (x) -> pdf(result.posteriors[:r], x), label = "q(r)", fill = 0, fillalpha = 0.2)
        p1 = vline!([realr], label = "real r")

        p2 = plot(0.0:0.01:1.0, (x) -> pdf(result.posteriors[:t], x), label = "q(t)", fill = 0, fillalpha = 0.2)
        p2 = vline!([realt], label = "real t")

        return plot(p1, p2)
    end
end

@testitem "Sunspot dataset" begin
    using StableRNGs, ExponentialFamilyProjection, BayesBase, Distributions, Plots

    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    # Here is only part of the dataset to speedup the test a bit
    dataset = [
        55,
        154,
        215,
        193,
        191,
        119,
        98,
        45,
        20,
        7,
        54,
        201,
        269,
        262,
        225,
        159,
        76,
        53,
        40,
        15,
        22,
        67,
        133,
        150,
        149,
        148,
        94,
        98,
        54,
        49,
        22,
        18,
        39,
        131,
        220,
        219,
        199,
        162,
        91,
        60,
        21,
        15,
        34,
        123,
        211,
        192,
        203,
        133,
        76,
        45,
        25,
        12,
        29,
        88,
        136,
        174,
        170,
        164,
        99,
        65,
        46,
        25,
        13,
        4,
        5,
        25,
        81,
        84,
        94,
        113,
        70,
        40,
        22,
        7,
        4,
        9,
        30
    ]

    @model function gamma_ssm(y)
        γ  ~ Gamma(α = 100.0, β = 0.1)
        z₀ ~ Gamma(α = 1.0, β = γ)
        for i in eachindex(y)
            z[i] ~ Gamma(α = z₀, β = γ)
            y[i] ~ Poisson(z[i])
            z₀ = z[i]
        end
    end

    init = @initialization begin
        q(γ)  = GammaShapeRate(1.0, 1.0)
        q(z₀) = GammaShapeRate(1.0, 1.0)
        q(z)  = GammaShapeRate(1.0, 1.0)
    end

    gamma_constraints = @constraints begin
        q(γ, z₀, z) = MeanField()
        q(γ)::ProjectedTo(Gamma)
        q(z₀)::ProjectedTo(Gamma)
        q(z)::ProjectedTo(Gamma)
    end

    result = infer(
        model = gamma_ssm(),
        iterations = 5,
        constraints = gamma_constraints,
        initialization = init,
        data = (y = dataset,),
        free_energy = true,
        options = (rulefallback = NodeFunctionRuleFallback(), limit_stack_depth = 500)
    )

    @test all(<(0), diff(result.free_energy))

    test_deviation = count(zip(result.posteriors[:z][end], dataset)) do (posterior_i, data_i)
        return mean(posterior_i) - 5std(posterior_i) < data_i < mean(posterior_i) + 5std(posterior_i)
    end

    @test test_deviation / length(dataset) > 0.8
    foreach(result.posteriors[:γ]) do posteriorγ
        @test all(p -> !isnan(p) && !isinf(p), params(posteriorγ))
    end
    foreach(result.posteriors[:z]) do posteriorsz
        foreach(posteriorsz) do posteriorz
            @test all(p -> !isnan(p) && !isinf(p), params(posteriorz))
        end
    end

    @test_plot "projection" "sunspot" begin
        p1 = plot(mean.(result.posteriors[:z][end]), ribbon = std.(result.posteriors[:z][end]))
        p1 = scatter!(p1, dataset)

        p2 = plot(result.free_energy)

        p3 = plot(0.0:0.01:2.0, (x) -> pdf(result.posteriors[:γ][end], x), fill = 0, fillalpha = 0.1)
        p4 = plot(0.0:0.1:40.0, (x) -> pdf(result.posteriors[:z₀][end], x), fill = 0, fillalpha = 0.1)

        return plot(p1, p2, p3, p4)
    end
end

@testitem "Inference with delta node and CVI projection" begin
    using StableRNGs, ExponentialFamilyProjection, ReactiveMP, Distributions, Plots

    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)

    @test !isnothing(ext)

    using .ext

    foo(x) = cos(x)
    bar(x) = exp(x) + x

    @model function iid_with_delta_transforms(y)
        a ~ Beta(1, 1)
        b ~ Beta(2, 2)

        mean := foo(a)
        precision := bar(b)

        y .~ Normal(mean = mean, precision = precision)
    end

    @constraints function iid_with_delta_transforms_constraints()
        q(a, b, mean, precision) = q(a)q(b)q(mean)q(precision)
        q(a)::ProjectedTo(Beta)
        q(b)::ProjectedTo(Beta)
        q(mean)::ProjectedTo(NormalMeanVariance)
        q(precision)::ProjectedTo(Gamma)
    end

    @meta function iid_with_delta_transforms_meta()
        foo() -> CVIProjection()
        bar() -> CVIProjection()
    end

    @initialization function iid_with_delta_transforms_initialization()
        q(a) = Beta(1, 1)
        q(b) = Beta(1, 1)
        q(mean) = NormalMeanVariance(0.5, 1)
        q(precision) = Gamma(1, 1)
    end

    function run_experiment(ra, rb)
        rmean = foo(ra)
        rprecision = bar(rb)

        rng = StableRNG(42)
        n = 1000
        y = rand(rng, NormalMeanPrecision(rmean, rprecision), n)

        result = infer(
            model = iid_with_delta_transforms(),
            data = (y = y,),
            constraints = iid_with_delta_transforms_constraints(),
            meta = iid_with_delta_transforms_meta(),
            initialization = iid_with_delta_transforms_initialization(),
            iterations = 2,
            returnvars = KeepLast(),
            free_energy = true
        )

        return ra, rb, rmean, rprecision, y, result
    end

    for ra in (0.23, 0.64), rb in (0.73, 0.13)
        ra, rb, rmean, rprecision, y, result = run_experiment(ra, rb)

        @test all(<(0), diff(result.free_energy))
        @test mean(result.posteriors[:a]) ≈ ra atol = 1e-1
        @test mean(result.posteriors[:b]) ≈ rb atol = 1e-1
        @test mean(result.posteriors[:mean]) ≈ rmean atol = 1e-1
        @test mean(result.posteriors[:precision]) ≈ rprecision atol = 2e-1
    end

    @test_plot "projection" "iid_delta" begin
        ra, rb, rmean, rprecision, y, result = run_experiment(0.22, 0.88)

        p1 = plot(0.0:0.01:1.0, (x) -> pdf(result.posteriors[:a], x), label = "inferred a", fill = 0, fillalpha = 0.2)
        p1 = vline!([ ra ], label = "real a")

        p2 = plot(0.0:0.01:1.0, (x) -> pdf(result.posteriors[:b], x), label = "inferred b", fill = 0, fillalpha = 0.2)
        p2 = vline!([ rb ], label = "real b")

        p3 = plot(0.0:0.01:5.0, (x) -> pdf(result.posteriors[:mean], x), label = "inferred mean", fill = 0, fillalpha = 0.2)
        p3 = vline!([ rmean ], label = "real mean")

        p4 = plot(0.0:0.01:5.0, (x) -> pdf(result.posteriors[:precision], x), label = "inferred precision", fill = 0, fillalpha = 0.2)
        p4 = vline!([ rprecision ], label = "real precision")

        plot(p1, p2, p3, p4)
    end
end