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
        return a + b
    end

    init = @initialization begin
        q(θ) = Beta(1.0, 1.0)
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
    @test getreturnval(result.model) === 9.0

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
        @test getreturnval(result_for_specific_p.model) === 2.0

        # Here we also double check the streaming version on this simple case
        # In the test we keep a small history of the size `1` since we are only interested in the last value
        # Additionally we subscribe on the posteriors to save the last value just to double check
        result_for_streaming_version = infer(
            model = beta_bernoulli_streaming(),
            data = (y = data_for_specific_p,),
            autoupdates = autoupdates,
            initialization = init,
            keephistory = 1, # Only keep the last one
            autostart = false
        )

        streaming_saved_result = Ref{Any}(nothing)
        streaming_analytical_i = Ref{Int}(0)
        streaming_analytical_a = Ref{Float64}(1.0) # The prior is `Beta(1.0, 1.0)`
        streaming_analytical_b = Ref{Float64}(1.0) # Thus we start with `a` and `b` equal to `1.0`
        streaming_analytical_check = Ref{Bool}(true)
        # During posteriors updates we double check that the result is consistent with the analytical solution
        streaming_subscription = subscribe!(
            result_for_streaming_version.posteriors[:θ], (new_posterior) -> begin
                streaming_analytical_i[] = streaming_analytical_i[] + 1
                streaming_analytical_observation = data_for_specific_p[streaming_analytical_i[]]

                # We know the analytical solution for the Beta-Bernoulli model
                # The `a` parameter should be incremented by `1` if the observation is `1`
                # The `b` parameter should be incremented by `1` if the observation is `0`
                streaming_analytical_a[] += isone(streaming_analytical_observation)
                streaming_analytical_b[] += iszero(streaming_analytical_observation)

                # We fetch the parameters of the posterior given from RxInfer inference engine
                new_posterior_a, new_posterior_b = params(new_posterior)

                if new_posterior_a !== streaming_analytical_a[] || new_posterior_b !== streaming_analytical_b[]
                    # If the intermediate result does not match the analytical solution we set the flag to `false`
                    # We avoid doing `@test` here since it generates hundreds of thousands of tests
                    streaming_analytical_check[] = false
                end

                # Save for later checking
                streaming_saved_result[] = new_posterior
            end
        )

        # `autostart` was set to false
        @test streaming_saved_result[] === nothing
        @test streaming_analytical_i[] === 0

        RxInfer.start(result_for_streaming_version)

        @test streaming_analytical_check[]
        @test streaming_analytical_i[] === n

        # Test that the result is the same for the streaming version
        @test result_for_specific_p.posteriors[:θ] == result_for_streaming_version.history[:θ][1]
        @test result_for_specific_p.posteriors[:θ] == streaming_saved_result[]
    end

    init = @initialization function beta_bernoulli_init(b)
        q(θ) = b
        μ(θ) = b
    end

    # In this model the result should not depend on the initial marginals or messages
    # But it should run anyway
    for θinit in (Beta(1.0, 1.0), Beta(2.0, 2.0), Beta(2.0, 7.0), Beta(7.0, 2.0))
        result_with_init = infer(model = beta_bernoulli(a = 2.0, b = 7.0), data = (y = dataset,), iterations = 10, initialization = beta_bernoulli_init(θinit), free_energy = true)

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
