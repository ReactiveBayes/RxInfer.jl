@testitem "`+` with any number of terms, constants among them" begin
    using BayesBase, Distributions

    # `a + b + c` is one `+` node; with y ~ N(s, 1/2) observed, the model is a tree and the free
    # energy is -log p(y), s ~ N(Σμ, Σv) a priori.
    @model function three_terms(y)
        a ~ Normal(mean = 0.0, variance = 1.0)
        b ~ Normal(mean = 1.0, variance = 2.0)
        c ~ Normal(mean = -2.0, variance = 0.5)
        s := a + b + c
        y ~ Normal(mean = s, variance = 0.5)
    end
    result = infer(model = three_terms(), data = (y = 2.0,), free_energy = true)
    @test only(result.free_energy) ≈ -logpdf(Normal(-1.0, 2.0), 2.0)
    @test mean(result.posteriors[:a]) ≈ 3 / 4 && var(result.posteriors[:a]) ≈ 3 / 4

    # A number among the terms is a constant of its own.
    @model function shifted(y)
        a ~ Normal(mean = 0.0, variance = 1.0)
        s := a + 1.0
        y ~ Normal(mean = s, variance = 1.0)
    end
    result = infer(model = shifted(), data = (y = 2.0,), free_energy = true)
    @test mean(result.posteriors[:a]) ≈ 0.5 && var(result.posteriors[:a]) ≈ 0.5
    @test only(result.free_energy) ≈ -logpdf(Normal(1.0, sqrt(2.0)), 2.0)

    # An observed sum: the terms lie on its plane, and the free energy is -log N(4 | 3, 4).
    @model function observed_sum(y)
        a ~ Normal(mean = 0.0, variance = 1.0)
        b ~ Normal(mean = 1.0, variance = 3.0)
        y := a + b + 2.0
    end
    result = infer(model = observed_sum(), data = (y = 4.0,), free_energy = true)
    @test mean(result.posteriors[:a]) ≈ 1 / 4 && var(result.posteriors[:a]) ≈ 3 / 4
    @test only(result.free_energy) ≈ -logpdf(Normal(3.0, 2.0), 4.0)
end
