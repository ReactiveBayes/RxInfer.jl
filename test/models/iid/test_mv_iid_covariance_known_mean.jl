module RxInferModelsMvIIDCovarianceKnownMeanTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

# Please use StableRNGs for random number generators

# `include(test/utiltests.jl)`
include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

@model function mv_iid_inverse_wishart_known_mean(mean, n, d)
    C ~ InverseWishart(d + 1, diageye(d))

    m = constvar(mean)
    y = datavar(Vector{Float64}, n)

    for i in 1:n
        y[i] ~ MvNormal(mean = m, covariance = C)
    end
end

function inference_mv_inverse_wishart_known_mean(mean, data, n, d)
    return inference(model = mv_iid_inverse_wishart_known_mean(mean, n, d), data = (y = data,), iterations = 10, returnvars = KeepLast(), free_energy = Float64)
end

@testset "Multivariate IID: Covariance parametrisation with known mean" begin

    ## Data creation
    rng = StableRNG(123)

    n = 1500
    d = 2

    m = rand(rng, d)
    L = randn(rng, d, d)
    C = L * L'

    data = rand(rng, MvNormalMeanCovariance(m, C), n) |> eachcol |> collect .|> collect

    ## Inference execution
    result_km = inference_mv_inverse_wishart_known_mean(m, data, n, d)

    ## Test inference results
    @test isapprox(mean(result_km.posteriors[:C]), C, atol = 0.15)
    # Check that FE does not depend on iteration in the known mean cast
    @test all(==(first(result_km.free_energy)), result_km.free_energy)

    @test_plot "models" "iid_mv_covariance_known_mean" begin
        X = range(-5, 5, length = 200)
        Y = range(-5, 5, length = 200)

        p = plot(title = "MvIID experiment / Covariance parametrisation with known mean")
        p = contour!(p, X, Y, (x, y) -> pdf(MvNormalMeanCovariance(m, mean(result_km.posteriors[:C])), [x, y]), label = "Estimated")
        p = contour!(p, X, Y, (x, y) -> pdf(MvNormalMeanCovariance(m, C), [x, y]), label = "Real")
    end

    @test_benchmark "models" "iid_mv_covariance_known_mean" inference_mv_inverse_wishart_known_mean($m, $data, $n, $d)
end

end
