@testitem "Coin Toss Model" begin
    using StableRNGs
    using Distributions

    n = 5000  # Number of coin flips
    p = 0.75 # Bias of a coin

    distribution = Bernoulli(p)
    dataset      = float.(rand(StableRNG(42), Bernoulli(p), n))

    @model function coin_model(y, a, b)
        θ ~ Beta(a, b)
        for i in eachindex(y)
            y[i] ~ Bernoulli(θ)
        end
    end

    # A simple execution of the inference, here the model is simple enough to be solved exactly
    # But we test for several iterations to make sure that the result is equal acros the iterationsl
    result = infer(model = coin_model(a = 2.0, b = 7.0), data = (y = dataset,), iterations = 10, free_energy = true)

    @test allequal(result.posteriors[:θ])
    @test allequal(result.free_energy)
    @test length(result.free_energy) === 10
    @test all(v -> v ≈ 2828.0533343622483, result.free_energy)
    @test mean(result.posteriors[:θ][end]) ≈ p atol = 1e-2

    # Double check that Free Energy computes properly even with no iterations specified (not equal to saying `iterations = 1`)
    result_with_no_iterations = infer(model = coin_model(a = 2.0, b = 7.0), data = (y = dataset,), free_energy = true)
    @test length(result_with_no_iterations.free_energy) === 1
    @test all(v -> v ≈ 2828.0533343622483, result_with_no_iterations.free_energy)

    # Double check for different values of `p`
    for p in (0.75, 0.5, 0.25), n in (5_000, 10_000)
        result_for_different_p = infer(model = coin_model(a = 1.0, b = 1.0), data = (y = float.(rand(StableRNG(123), Bernoulli(p), n)),))
        @test mean(result_for_different_p.posteriors[:θ]) ≈ p atol = 1e-2
    end

    # In this model the result should not depend on the initial marginals or messages
    # But it should run anyway
    for θinit in (Beta(1.0, 1.0), Beta(2.0, 2.0), Beta(2.0, 7.0), Beta(7.0, 2.0))
        result_with_init = infer(model = coin_model(a = 2.0, b = 7.0), data = (y = dataset,), iterations = 10, initmarginals = (θ = θinit,), initmessages = (θ = θinit,), free_energy = true)

        @test all(t -> t[1] == t[2], Iterators.zip(result.posteriors[:θ], result_with_init.posteriors[:θ]))
        @test all(t -> t[1] == t[2], Iterators.zip(result.free_energy, result_with_init.free_energy))
    end

    # Free energy allows for a specific type for the result of the free energy to make it a little more efficient
    for T in (BigFloat, Float64, Float32)
        result_with_specific_free_energy_type = infer(model = coin_model(a = 2.0, b = 7.0), data = (y = dataset,), iterations = 10, free_energy = T)

        @test all(t -> t[1] == t[2], Iterators.zip(result.posteriors[:θ], result_with_specific_free_energy_type.posteriors[:θ]))
        @test all(t -> t[1] ≈ t[2], Iterators.zip(result.free_energy, result_with_specific_free_energy_type.free_energy))
        @test all(value -> value isa T, result_with_specific_free_energy_type.free_energy)
    end
end
