@testitem "datavar, randomvar or constvars are disallowed in the new versions" begin
    dataset = float.(rand(Bernoulli(0.5), 50))

    @model function coin_model_1_error_datavar(n, a, b, y)
        θ ~ Beta(a, b)
        y = datavar(Float64, n)
        for i in 1:length(y)
            y[i] ~ Bernoulli(θ)
        end
    end

    @model function coin_model_2_error_constvar(a, b, y)
        a_ = constvar(a)
        b_ = constvar(b)
        θ ~ Beta(a_, b_)
        for i in eachindex(y)
            y[i] ~ Bernoulli(θ)
        end
    end

    @model function coin_model_3_error_randomvar(a, b, y)
        θ = randomvar()
        θ ~ Beta(a, b)
        for i in eachindex(y)
            y[i] ~ Bernoulli(θ)
        end
    end

    @test_throws "`datavar`, `constvar` and `randomvar` syntax has been removed" infer(model = coin_model_1_error_datavar(n = 50, a = 2.0, b = 7.0), data = (y = dataset,))

    @test_throws "`datavar`, `constvar` and `randomvar` syntax has been removed" infer(model = coin_model_2_error_constvar(a = 2.0, b = 7.0), data = (y = dataset,))

    @test_throws "`datavar`, `constvar` and `randomvar` syntax has been removed" infer(model = coin_model_3_error_randomvar(a = 2.0, b = 7.0), data = (y = dataset,))
end

@testitem "Priors in arguments" begin
    using StableRNGs
    using Distributions

    @model function coin_model_priors(y, prior)
        θ ~ prior
        for i in eachindex(y)
            y[i] ~ Bernoulli(θ)
        end
    end

    rng  = StableRNG(42)
    n    = 50
    data = float.(rand(rng, Bernoulli(0.75), n))

    testsets = [(prior = Beta(4.0, 8.0), answer = Beta(43.0, 19.0)), (prior = Beta(54.0, 1.0), answer = Beta(93.0, 12.0)), (prior = Beta(1.0, 12.0), answer = Beta(40.0, 23.0))]

    for ts in testsets
        @test infer(model = coin_model_priors(prior = ts[:prior]), data = (y = data,)).posteriors[:θ] == ts[:answer]
    end

    struct ArbitraryNonExistingDistribution <: UnivariateDistribution end

    @test_throws "It is not possible to materialize a factor `ArbitraryNonExistingDistribution()` without prespecified interfaces." infer(
        model = coin_model_priors(prior = ArbitraryNonExistingDistribution()), data = (y = data,)
    )
end