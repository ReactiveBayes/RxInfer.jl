@testitem "Multivariate IID: Covariance parametrisation" begin
    using StableRNGs, Plots, BenchmarkTools

    # `include(test/utiltests.jl)`
    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    @model function mv_iid_inverse_wishart(n, d)
        m ~ MvNormal(mean = zeros(d), precision = 100 * diageye(d))
        C ~ InverseWishart(d + 1, diageye(d))

        y = datavar(Vector{Float64}, n)

        for i in 1:n
            y[i] ~ MvNormal(mean = m, covariance = C)
        end
    end

    @constraints function constraints_mv_iid_inverse_wishart()
        q(m, C) = q(m)q(C)
    end

    function inference_mv_inverse_wishart(data, n, d)
        return infer(
            model = mv_iid_inverse_wishart(n, d),
            data = (y = data,),
            constraints = constraints_mv_iid_inverse_wishart(),
            initmarginals = (m = vague(MvNormalMeanCovariance, d), C = vague(InverseWishart, d)),
            returnvars = KeepLast(),
            iterations = 10,
            free_energy = Float64
        )
    end

    ## Data creation
    rng = StableRNG(123)

    n = 1500
    d = 2

    m = rand(rng, d)
    L = randn(rng, d, d)
    C = L * L'

    data = rand(rng, MvNormalMeanCovariance(m, C), n) |> eachcol |> collect .|> collect

    ## Inference execution
    result = inference_mv_inverse_wishart(data, n, d)

    ## Test inference results
    @test isapprox(mean(result.posteriors[:m]), m, atol = 0.07)
    @test isapprox(mean(result.posteriors[:C]), C, atol = 0.15)
    @test all(<(0), filter(e -> abs(e) > 1e-10, diff(result.free_energy)))

    @test_plot "models" "iid_mv_covariance" begin
        X = range(-5, 5, length = 200)
        Y = range(-5, 5, length = 200)

        p = plot(title = "MvIID experiment / Covariance parametrisation")
        p = contour!(p, X, Y, (x, y) -> pdf(MvNormalMeanCovariance(mean(result.posteriors[:m]), mean(result.posteriors[:C])), [x, y]), label = "Estimated")
        p = contour!(p, X, Y, (x, y) -> pdf(MvNormalMeanCovariance(m, C), [x, y]), label = "Real")
    end

    @test_benchmark "models" "iid_inverse_wishart" inference_mv_inverse_wishart($data, $n, $d)
end
