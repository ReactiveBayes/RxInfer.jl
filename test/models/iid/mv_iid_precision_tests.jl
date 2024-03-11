@testitem "Multivariate IID: Precision parametrisation" begin
    using StableRNGs, BenchmarkTools, Plots
    # Please use StableRNGs for random number generators

    # `include(test/utiltests.jl)`
    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    ## Model and constraints definition

    @model function mv_iid_wishart(y, d)
        m ~ MvNormal(μ = zeros(d), Λ = 100 * diageye(d))
        P ~ Wishart(d + 1, diageye(d))
        for i in eachindex(y)
            y[i] ~ MvNormal(μ = m, Λ = P)
        end
    end

    constraints_mv_iid_wishart = @constraints begin
        q(m, P) = q(m)q(P)
    end

    ## Inference definition

    function inference_mv_wishart(data, d)
        return infer(
            model = mv_iid_wishart(d = d),
            data = (y = data,),
            constraints = constraints_mv_iid_wishart,
            initmarginals = (m = vague(MvNormalMeanCovariance, d), P = vague(Wishart, d)),
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
    P = inv(C)

    data = rand(rng, MvNormalMeanPrecision(m, P), n) |> eachcol |> collect .|> collect

    ## Inference execution
    result = inference_mv_wishart(data, d)

    ## Test inference results
    @test isapprox(mean(result.posteriors[:m]), m, atol = 0.05)
    @test isapprox(mean(result.posteriors[:P]), P, atol = 0.07)
    @test all(<(0), filter(e -> abs(e) > 1e-10, diff(result.free_energy)))

    @test_plot "models" "iid_mv_precision" begin
        X = range(-5, 5, length = 200)
        Y = range(-5, 5, length = 200)

        p = plot(title = "MvIID experiment / Precision parametrisation")
        p = contour!(p, X, Y, (x, y) -> pdf(MvNormalMeanPrecision(mean(result.posteriors[:m]), mean(result.posteriors[:P])), [x, y]), label = "Estimated")
        p = contour!(p, X, Y, (x, y) -> pdf(MvNormalMeanPrecision(m, P), [x, y]), label = "Real")
    end

    @test_benchmark "models" "iid_mv_wishart" inference_mv_wishart($data, $d)
end
