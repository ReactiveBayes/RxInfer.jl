@testitem "aliases for `Gamma` family of distributions" begin
    @model function gamma_aliases(y)
        # shape-scale parametrization
        γ[1] ~ Gamma(shape = 1.0, scale = 1.0)
        γ[2] ~ Gamma(a = 1.0, θ = 1.0)
        γ[3] ~ Gamma(α = 1.0, β⁻¹ = 1.0)

        # shape-rate parametrization
        γ[4] ~ Gamma(shape = 1.0, rate = 1.0)
        γ[5] ~ Gamma(a = 1.0, θ⁻¹ = 1.0)
        γ[6] ~ Gamma(α = 1.0, β = 1.0)

        x[1] ~ Normal(μ = 1.0, σ⁻² = γ[1])
        x[2] ~ Normal(μ = 1.0, σ⁻² = γ[2])
        x[3] ~ Normal(μ = 1.0, σ⁻² = γ[3])
        x[4] ~ Normal(μ = 1.0, σ⁻² = γ[4])
        x[5] ~ Normal(μ = 1.0, σ⁻² = γ[5])
        x[6] ~ Normal(μ = 1.0, σ⁻² = γ[6])

        s ~ x[1] + x[2] + x[3] + x[4] + x[5] + x[6]
        y ~ Normal(μ = s, σ² = 1.0)
    end

    constraints = @constraints begin
        q(x, γ) = q(x)q(γ)
    end

    init = @initialization begin
        q(x) = vague(NormalMeanVariance)
        q(γ) = vague(GammaShapeRate)
    end

    results = infer(model = gamma_aliases(), data = (y = 10.0,), constraints = constraints, init = init, free_energy = true)
    # Here we simply test that it ran and gave some output 
    @test mean(results.posteriors[:s]) ≈ 9.20000000000032
    @test first(results.free_energy) ≈ 1.3847462395606698
end
