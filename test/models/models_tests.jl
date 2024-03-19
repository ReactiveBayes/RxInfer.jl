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

@testitem "A `Distribution` object as priors in arguments" begin
    using StableRNGs
    using Distributions

    @model function beta_bernoulli_priors(y, prior)
        θ ~ prior
        for i in eachindex(y)
            y[i] ~ Bernoulli(θ)
        end
    end

    @model function beta_bernoulli_params(y, a, b)
        θ ~ Beta(a, b)
        for i in eachindex(y)
            y[i] ~ Bernoulli(θ)
        end
    end

    @testset "Beta-Bernoulli model" begin
        for seed in (123, 456), n in (50, 100)
            rng  = StableRNG(seed)
            data = float.(rand(rng, Bernoulli(0.75), n))

            count_trues = count(isone, data)
            count_falses = count(iszero, data)

            testsets = [
                (prior = Beta(4.0, 8.0), answer = Beta(4 + count_trues, 8 + count_falses)),
                (prior = Beta(54.0, 1.0), answer = Beta(54 + count_trues, 1 + count_falses)),
                (prior = Beta(1.0, 12.0), answer = Beta(1 + count_trues, 12 + count_falses))
            ]

            for ts in testsets
                result_with_prior_as_input = infer(
                    model = beta_bernoulli_priors(prior = ts[:prior]), returnvars = KeepLast(), data = (y = data,), iterations = 10, free_energy = true
                )
                result_with_params_as_input = infer(
                    model = beta_bernoulli_params(a = ts[:prior].α, b = ts[:prior].β), returnvars = KeepLast(), data = (y = data,), iterations = 10, free_energy = true
                )

                @test result_with_prior_as_input.posteriors[:θ] == ts[:answer]
                @test result_with_params_as_input.posteriors[:θ] == ts[:answer]
                @test all(result_with_prior_as_input.free_energy .≈ result_with_params_as_input.free_energy)
            end
        end
    end

    @model function iid_gaussians_priors(y, prior_for_μ, prior_for_τ)
        μ ~ prior_for_μ
        τ ~ prior_for_τ
        for i in eachindex(y)
            y[i] ~ Normal(mean = μ, precision = τ)
        end
    end

    @model function iid_gaussians_params(y, mean, variance, shape, scale)
        μ ~ Normal(mean = mean, variance = variance)
        τ ~ Gamma(shape = shape, scale = scale)
        for i in eachindex(y)
            y[i] ~ Normal(mean = μ, precision = τ)
        end
    end

    @testset "IID Gaussian model" begin
        for seed in (123, 456), n in (50, 100)
            rng  = StableRNG(seed)
            data = float.(rand(rng, Normal(0.75, 10.0), n))

            testsets = [
                (prior_for_μ = NormalMeanVariance(4.0, 8.0), prior_for_τ = Gamma(4.0, 8.0)),
                (prior_for_μ = NormalMeanVariance(54.0, 1.0), prior_for_τ = Gamma(54.0, 1.0)),
                (prior_for_μ = NormalMeanVariance(1.0, 12.0), prior_for_τ = Gamma(1.0, 12.0))
            ]

            initmarginals = (μ = NormalMeanVariance(0.0, 1.0), τ = Gamma(1.0, 1.0))

            for ts in testsets
                result_with_prior_as_input = infer(
                    model = iid_gaussians_priors(prior_for_μ = ts[:prior_for_μ], prior_for_τ = ts[:prior_for_τ]),
                    returnvars = KeepLast(),
                    initmarginals = initmarginals,
                    constraints = MeanField(),
                    data = (y = data,),
                    iterations = 10,
                    free_energy = true
                )
                result_with_params_as_input = infer(
                    model = iid_gaussians_params(mean = mean(ts[:prior_for_μ]), variance = var(ts[:prior_for_μ]), shape = shape(ts[:prior_for_τ]), scale = scale(ts[:prior_for_τ])),
                    returnvars = KeepLast(),
                    initmarginals = initmarginals,
                    constraints = MeanField(),
                    data = (y = data,),
                    iterations = 10,
                    free_energy = true
                )

                @test result_with_prior_as_input.posteriors[:μ] == result_with_params_as_input.posteriors[:μ]
                @test result_with_prior_as_input.posteriors[:τ] == result_with_params_as_input.posteriors[:τ]
                @test all(result_with_prior_as_input.free_energy .≈ result_with_params_as_input.free_energy)
            end
        end
    end
end

@testitem "Unknown object in the model specification should throw a user-friendly error" begin
    struct SomeArbitraryDistribution
        a::Float64
        b::Float64
    end

    @model function a_model_with_unknown_distribution(y)
        θ ~ SomeArbitraryDistribution(1.0, 2.0)
        for i in eachindex(y)
            y[i] ~ Bernoulli(θ)
        end
    end

    @test_throws "`$(SomeArbitraryDistribution)` cannot be used as a factor node" infer(model = a_model_with_unknown_distribution(), data = (y = ones(3),)).posteriors[:θ]
end