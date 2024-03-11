@testitem "Multivariate IID: Precision parametrisation with known mean" begin
    using StableRNGs, BenchmarkTools, Plots
    # Please use StableRNGs for random number generators

    # `include(test/utiltests.jl)`
    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    ## Model and constraints definition

    @model function mv_iid_wishart_known_mean(m, y, d)
        P ~ Wishart(d + 1, diageye(d))
        for i in eachindex(y)
            y[i] ~ MvNormal(μ = m, Λ = P)
        end
    end

    function inference_mv_wishart_known_mean(mean, data, d)
        return infer(model = mv_iid_wishart_known_mean(m = mean, d = d), data = (y = data,), iterations = 10, returnvars = KeepLast(), free_energy = Float64)
    end

    ## Data creation
    rng = StableRNG(123)

    n = 1500
    d = 2

    m = rand(rng, d)
    L = randn(rng, d, d)
    C = L * L'
    P = inv(C)

    data = rand(rng, MvNormalMeanPrecision(m, P), n) |> eachcol |> collect .|> collect

    ## Inference execution
    result_km = inference_mv_wishart_known_mean(m, data, d)

    ## Test inference results
    @test isapprox(mean(result_km.posteriors[:P]), P, atol = 0.07)
    # Check that FE does not depend on iteration in the known mean cast
    @test all(==(first(result_km.free_energy)), result_km.free_energy)

    @test_plot "models" "iid_mv_wishart_known_mean" begin
        X = range(-5, 5, length = 200)
        Y = range(-5, 5, length = 200)

        p = plot(title = "MvIID experiment / Precision parametrisation (known mean)")
        p = contour!(p, X, Y, (x, y) -> pdf(MvNormalMeanPrecision(m, mean(result_km.posteriors[:P])), [x, y]), label = "Estimated")
        p = contour!(p, X, Y, (x, y) -> pdf(MvNormalMeanPrecision(m, P), [x, y]), label = "Real")
    end

    @test_benchmark "models" "iid_wishart_known_mean" inference_mv_wishart_known_mean($m, $data, $n, $d)
end
