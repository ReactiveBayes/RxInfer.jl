@testitem "aliases for binary operations" begin
    @model function binary_aliases(y)
        x1 ~ Bernoulli(0.5)
        x2 ~ Bernoulli(0.5)
        x3 ~ Bernoulli(0.5)
        x4 ~ Bernoulli(0.5)

        x ~ x1 -> x2 && x3 || ¬x4

        x ~ Bernoulli(y)
    end

    function binary_aliases_inference()
        return inference(model = binary_aliases(), data = (y = 0.5,), free_energy = true)
    end
    results = binary_aliases_inference()
    # Here we simply test that it ran and gave some output 
    @test mean(results.posteriors[:x1]) ≈ 0.5
    @test first(results.free_energy) ≈ 0.6931471805599454
end
