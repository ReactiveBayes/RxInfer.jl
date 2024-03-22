@testitem "Beta-Bernoulli model" begin
    using StableRNGs
    using Distributions

    n = 5000  # Number of coin flips
    p = 0.75 # Bias of a coin

    distribution = Bernoulli(p)
    dataset      = float.(rand(StableRNG(42), distribution, n))

    @model function beta_bernoulli(y, a, b)
        θ ~ Beta(a, b)
        for i in eachindex(y)
            y[i] ~ Bernoulli(θ)
        end
    end

    # Check the streaming version of the same model
    @model function beta_bernoulli_streaming(y, a, b)
        θ ~ Beta(a, b)
        y ~ Bernoulli(θ)
    end

    autoupdates = @autoupdates begin
        # The `a` and `b` parameters are being updated in the streaming fashion
        # Each new posterior for `q(θ)` defines the new values for `a` and `b`
        a, b = params(q(θ))
    end

    # The initial value for `θ` in the `@autoupdates` has not been specified, the `initmarginals` should be used
    # The `beta_bernoulli_streaming` with `initmarginals` is being tested later 
    @test_throws "The initial value for `θ` in the `@autoupdates` has not been specified" infer(
        model = beta_bernoulli_streaming(), data = (y = dataset,), autoupdates = autoupdates
    )

    # A simple execution of the inference, here the model is simple enough to be solved exactly
    # But we test for several iterations to make sure that the result is equal acros the iterationsl
    result = infer(model = beta_bernoulli(a = 2.0, b = 7.0), data = (y = dataset,), iterations = 10, free_energy = true)

    @test allequal(result.posteriors[:θ])
    @test allequal(result.free_energy)
    @test length(result.free_energy) === 10
    @test all(v -> v ≈ 2828.0533343622483, result.free_energy)
    @test mean(result.posteriors[:θ][end]) ≈ p atol = 1e-2

    # Double check that Free Energy computes properly even with no iterations specified (not equal to saying `iterations = 1`)
    result_with_no_iterations = infer(model = beta_bernoulli(a = 2.0, b = 7.0), data = (y = dataset,), free_energy = true)
    @test length(result_with_no_iterations.free_energy) === 1
    @test all(v -> v ≈ 2828.0533343622483, result_with_no_iterations.free_energy)

    # Double check that Free Energy is exactly the same for `MeanField` constraints because the model is simple enough and has only one variable
    result_mean_field = infer(model = beta_bernoulli(a = 2.0, b = 7.0), constraints = MeanField(), data = (y = dataset,), iterations = 10, free_energy = true)
    @test length(result_mean_field.free_energy) === 10
    @test all(v -> v ≈ 2828.0533343622483, result_mean_field.free_energy)

    # Double check for different values of `p`
    for p in (0.75, 0.5, 0.25), n in (5_000, 10_000)
        local data_for_specific_p = float.(rand(StableRNG(123), Bernoulli(p), n)) 

        result_for_specific_p = infer(model = beta_bernoulli(a = 1.0, b = 1.0), data = (y = data_for_specific_p,))
        @test mean(result_for_specific_p.posteriors[:θ]) ≈ p atol = 1e-2

        result_for_streaming_version = infer(
            model = beta_bernoulli_streaming(),
            data = (y = data_for_specific_p,), 
            autoupdates = autoupdates,
            initmarginals = (θ = Beta(1.0, 1.0), ),
            keephistory = 1, # Only keep the last one
        )

        # Test that the result is the same for the streaming version
        @test result_for_specific_p.posteriors[:θ] == result_for_streaming_version.history[:θ][1]
    end

    # In this model the result should not depend on the initial marginals or messages
    # But it should run anyway
    for θinit in (Beta(1.0, 1.0), Beta(2.0, 2.0), Beta(2.0, 7.0), Beta(7.0, 2.0))
        result_with_init = infer(
            model = beta_bernoulli(a = 2.0, b = 7.0), data = (y = dataset,), iterations = 10, initmarginals = (θ = θinit,), initmessages = (θ = θinit,), free_energy = true
        )

        @test all(t -> t[1] == t[2], Iterators.zip(result.posteriors[:θ], result_with_init.posteriors[:θ]))
        @test all(t -> t[1] == t[2], Iterators.zip(result.free_energy, result_with_init.free_energy))
    end

    # Free energy allows for a specific type for the result of the free energy to make it a little more efficient
    for T in (BigFloat, Float64, Float32)
        result_with_specific_free_energy_type = infer(model = beta_bernoulli(a = 2.0, b = 7.0), data = (y = dataset,), iterations = 10, free_energy = T)

        @test all(t -> t[1] == t[2], Iterators.zip(result.posteriors[:θ], result_with_specific_free_energy_type.posteriors[:θ]))
        @test all(t -> t[1] ≈ t[2], Iterators.zip(result.free_energy, result_with_specific_free_energy_type.free_energy))
        @test all(value -> value isa T, result_with_specific_free_energy_type.free_energy)
    end
end
