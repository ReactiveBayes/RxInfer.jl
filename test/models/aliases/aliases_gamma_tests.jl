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

    results = infer(model = gamma_aliases(), data = (y = 10.0,), constraints = constraints, iterations = 100, initialization = init, free_energy = true)

    # Here we simply test that it ran and gave some output 
    @test mean(results.posteriors[:s][end]) ≈ 9.468846338832027
    @test first(results.free_energy[end]) ≈ 4.385584096993327
    @test all(<=(1e-14), diff(results.free_energy)) # it oscilates a bit at the end, but all should be less or equal to zero
end

@testitem "`Gamma` by itself cannot be used as a node" begin
    using Logging

    @model function gamma_by_itself(d)
        d ~ Gamma(1.0, 1.0)
    end

    io = IOBuffer()

    Logging.with_logger(Logging.SimpleLogger(io)) do
        infer(model = gamma_by_itself(), data = (d = 1.0,))
    end

    @test occursin("'Gamma' and 'GammaShapeScale' without keywords are constructed with parameters (Shape, Scale)", String(take!(io)))
end
