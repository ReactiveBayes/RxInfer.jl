module RxInferModelsMLGSSMTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

# `include(test/utiltests.jl)`
include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

@model function univariate_lgssm_model(n, x0, c_, P_)
    x_prior ~ Normal(mean = mean(x0), var = var(x0))

    x = randomvar(n)
    c = constvar(c_)
    P = constvar(P_)
    y = datavar(Float64, n)

    x_prev = x_prior

    for i in 1:n
        x[i] ~ x_prev + c
        y[i] ~ Normal(mean = x[i], var = P)
        x_prev = x[i]
    end
end

function univariate_lgssm_inference(data, x0, c, P)
    return inference(model = univariate_lgssm_model(length(data), x0, c, P), data = (y = data,), free_energy = true)
end

@testset "Univariate Linear Gaussian State Space Model" begin

    ## Data creation
    rng      = StableRNG(123)
    P        = 100.0
    n        = 500
    hidden   = collect(1:n)
    data     = hidden + rand(rng, Normal(0.0, sqrt(P)), n)
    x0_prior = NormalMeanVariance(0.0, 10000.0)

    ## Inference execution
    uresult = univariate_lgssm_inference(data, x0_prior, 1.0, P)
    x_estimated = uresult.posteriors[:x]
    fe = uresult.free_energy

    ## Test inference results
    @test length(x_estimated) === n
    @test all((mean.(x_estimated) .- 3 .* std.(x_estimated)) .< hidden .< (mean.(x_estimated) .+ 3 .* std.(x_estimated)))
    @test all(var.(x_estimated) .> 0.0)
    @test length(fe) === 1
    @test abs(last(fe) - 1854.297647) < 0.01

    ## Create output plots
    @test_plot "models" "ulgssm" begin
        subrange = 200:215
        m = mean.(x_estimated)[subrange]
        s = std.(x_estimated)[subrange]
        p = plot()
        p = plot!(subrange, m, ribbon = s, label = "Estimated signal")
        p = plot!(subrange, hidden[subrange], label = "Hidden signal")
        p = scatter!(subrange, data[subrange], label = "Observations")
        return p
    end

    @test_benchmark "models" "ulgssm" univariate_lgssm_inference($data, $x0_prior, 1.0, $P)
end

end
