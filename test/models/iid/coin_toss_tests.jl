@testitem "Coin Toss Model" begin
    using StableRNGs
    using Distributions

    n = 500  # Number of coin flips
    p = 0.75 # Bias of a coin

    distribution = Bernoulli(p)
    dataset      = float.(rand(StableRNG(42), Bernoulli(p), n))

    @model function coin_model(y)
        θ ~ Beta(2.0, 7.0)

        for i in eachindex(y)
            y[i] ~ Bernoulli(θ)
        end
    end

    result = infer(model = coin_model(), data  = (y = dataset,))

    @test mean(result.posteriors[:θ]) ≈ p atol = 0.01
end
