@testitem "Coin Toss Model" begin
    using StableRNGs
    using Distributions

    n = 5000  # Number of coin flips
    p = 0.75 # Bias of a coin

    distribution = Bernoulli(p)
    dataset      = float.(rand(StableRNG(42), Bernoulli(p), n))

    @model function coin_model(y)
        θ ~ Beta(2.0, 7.0)

        for i in eachindex(y)
            y[i] ~ Bernoulli(θ)
        end
    end

    result = infer(
        model = coin_model(),
        data = (y = dataset,), 
        iterations = 10,
        free_energy = true
    )

    @test allequal(result.posteriors[:θ])
    @test allequal(result.free_energy)
    @test all(v -> v ≈ 2828.0533343622483, result.free_energy)
    @test mean(result.posteriors[:θ][end]) ≈ p atol = 0.01

    # In this model the result should not depend on the initial marginals or messages
    # But it should run anyway
    result_with_init = infer(
        model = coin_model(), 
        data = (y = dataset,), 
        iterations = 10,
        initmarginals = (
            θ = Beta(1.0, 1.0),
        ),
        initmessages = (
            θ = Beta(1.0, 1.0),
        ),
        free_energy = true
    )

    @test all(t -> t[1] == t[2], Iterators.zip(result.posteriors[:θ], result_with_init.posteriors[:θ]))
    @test all(t -> t[1] == t[2], Iterators.zip(result.free_energy, result_with_init.free_energy))
end
