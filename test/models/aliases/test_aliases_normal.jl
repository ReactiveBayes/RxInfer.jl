module RxInferModelsAliasesTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, LinearAlgebra, StableRNGs

@model function normal_aliases()
    x1 ~ MvNormal(μ = zeros(2), Σ⁻¹ = diageye(2))
    x2 ~ MvNormal(μ = zeros(2), Λ = diageye(2))
    x3 ~ MvNormal(mean = zeros(2), W = diageye(2))
    x4 ~ MvNormal(μ = zeros(2), prec = diageye(2))
    x5 ~ MvNormal(m = zeros(2), precision = diageye(2))

    y1 ~ MvNormal(mean = zeros(2), Σ = diageye(2))
    y2 ~ MvNormal(m = zeros(2), Λ⁻¹ = diageye(2))
    y3 ~ MvNormal(μ = zeros(2), V = diageye(2))
    y4 ~ MvNormal(mean = zeros(2), cov = diageye(2))
    y5 ~ MvNormal(mean = zeros(2), covariance = diageye(2))

    x ~ x1 + x2 + x3 + x4 + x5
    y ~ y1 + y2 + y3 + y4 + y5

    r1 ~ Normal(μ = dot(x + y, ones(2)), τ = 1.0)
    r2 ~ Normal(m = r1, γ = 1.0)
    r3 ~ Normal(mean = r2, σ⁻² = 1.0)
    r4 ~ Normal(mean = r3, w = 1.0)
    r5 ~ Normal(mean = r4, p = 1.0)
    r6 ~ Normal(mean = r5, prec = 1.0)
    r7 ~ Normal(mean = r6, precision = 1.0)

    s1 ~ Normal(μ = r7, σ² = 1.0)
    s2 ~ Normal(m = s1, τ⁻¹ = 1.0)
    s3 ~ Normal(mean = s2, v = 1.0)
    s4 ~ Normal(mean = s3, var = 1.0)
    s5 ~ Normal(mean = s4, variance = 1.0)

    d = datavar(Float64)
    d ~ Normal(μ = s5, variance = 1.0)
end

function normal_aliases_inference() 
    return inference(
        model = normal_aliases(),
        data = (d = 1.0,),
        returnvars = (x1 = KeepLast(),),
        free_energy = true
    )
end

@testset "aliases for `Normal` family of distributions" begin
    result = normal_aliases_inference()
    # Here we simply test that it ran and gave some output 
    @test first(mean(result.posteriors[:x1])) ≈ 0.03030314940197132 
    @test first(result.free_energy) ≈ 2.6823450222125444
end

end
