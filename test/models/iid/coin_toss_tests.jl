@testitem "Coin Toss Model" begin
    
    using StableRNGs

    n = 500  # Number of coin flips
    p = 0.75 # Bias of a coin

    distribution = Bernoulli(p) 
    dataset      = float.(rand(StableRNG(42), Bernoulli(p), n))

    @model function coin_model(y, n)
        y = datavar(Float64, n)
        θ ~ Beta(2.0, 7.0)
        
        for i in 1:n
            y[i] ~ Bernoulli(θ)
        end

    end

    result = infer(
        model = coin_model(n = length(dataset)),
        data  = (y = dataset, )
    )

    @test mean(result.posteriors[:θ]) ≈ p atol=0.01
end

