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

    @model function coin_model_without_prior_for_check(y, a, b)
        θ ~ Beta(a, b)
        for i in eachindex(y)
            y[i] ~ Bernoulli(θ)
        end
    end

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
            result_with_prior_as_input = infer(model = coin_model_priors(prior = ts[:prior]), returnvars = KeepLast(), data = (y = data,), iterations = 10, free_energy = true)
            result_with_parameters_as_input = infer(
                model = coin_model_without_prior_for_check(a = ts[:prior].α, b = ts[:prior].β), returnvars = KeepLast(), data = (y = data,), iterations = 10, free_energy = true
            )

            @test result_with_prior_as_input.posteriors[:θ] == ts[:answer]
            @test result_with_parameters_as_input.posteriors[:θ] == ts[:answer]
            # `≈` here because the average energy computation is different for such nodes, basically it avoid creating 
            # nodes for constants `α` and `β`
            @test all(result_with_prior_as_input.free_energy .≈ result_with_parameters_as_input.free_energy)
        end
    end
end