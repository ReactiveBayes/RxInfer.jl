module RxInferModelsMvIIDTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

# Please use StableRNGs for random number generators

## Model and constraints definition
## -------------------------------------------- ##
@model function mv_iid_wishart(n, d)
    m ~ MvNormalMeanPrecision(zeros(d), 100 * diageye(d))
    P ~ Wishart(d + 1, diageye(d))

    y = datavar(Vector{Float64}, n)

    for i in 1:n
        y[i] ~ MvNormalMeanPrecision(m, P)
    end
end

@constraints function constraints_mv_iid_wishart()
    q(m, P) = q(m)q(P)
end
## -------------------------------------------- ##
@model function mv_iid_inverse_wishart(n, d)
    m ~ MvNormalMeanPrecision(zeros(d), 100 * diageye(d))
    C ~ InverseWishart(d + 1, diageye(d))

    y = datavar(Vector{Float64}, n)

    for i in 1:n
        y[i] ~ MvNormalMeanCovariance(m, C)
    end
end

@constraints function constraints_mv_iid_inverse_wishart()
    q(m, C) = q(m)q(C)
end
## -------------------------------------------- ##
@model function mv_iid_wishart_known_mean(mean, n, d)
    P ~ Wishart(d + 1, diageye(d))

    m = constvar(mean)
    y = datavar(Vector{Float64}, n)

    for i in 1:n
        y[i] ~ MvNormalMeanPrecision(m, P)
    end
end
## -------------------------------------------- ##
@model function mv_iid_inverse_wishart_known_mean(mean, n, d)
    C ~ InverseWishart(d + 1, diageye(d))

    m = constvar(mean)
    y = datavar(Vector{Float64}, n)

    for i in 1:n
        y[i] ~ MvNormalMeanCovariance(m, C)
    end
end
## -------------------------------------------- ##
## Inference definition
## -------------------------------------------- ##
function inference_mv_wishart(data, n, d)
    return inference(
        model = Model(mv_iid_wishart, n, d),
        data = (y = data,),
        constraints = constraints_mv_iid_wishart(),
        initmarginals = (
            m = vague(MvNormalMeanCovariance, d),
            P = vague(Wishart, d)
        ),
        returnvars = KeepLast(),
        iterations = 10,
        free_energy = Float64
    )
end
## -------------------------------------------- ##
function inference_mv_inverse_wishart(data, n, d)
    return inference(
        model = Model(mv_iid_inverse_wishart, n, d),
        data = (y = data,),
        constraints = constraints_mv_iid_inverse_wishart(),
        initmarginals = (
            m = vague(MvNormalMeanCovariance, d),
            C = vague(InverseWishart, d)
        ),
        returnvars = KeepLast(),
        iterations = 10,
        free_energy = Float64
    )
end
## -------------------------------------------- ##
function inference_mv_wishart_known_mean(mean, data, n, d)
    return inference(
        model = Model(mv_iid_wishart_known_mean, mean, n, d),
        data = (y = data,),
        iterations = 10,
        returnvars = KeepLast(),
        free_energy = Float64
    )
end
## -------------------------------------------- ##
function inference_mv_inverse_wishart_known_mean(mean, data, n, d)
    return inference(
        model = Model(mv_iid_inverse_wishart_known_mean, mean, n, d),
        data = (y = data,),
        iterations = 10,
        returnvars = KeepLast(),
        free_energy = Float64
    )
end
## -------------------------------------------- ##

@testset "Multivariate IID" begin
    @testset "Use case #1: Precision parametrisation" begin
        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
        rng = StableRNG(123)

        n = 1500
        d = 2

        m = rand(rng, d)
        L = randn(rng, d, d)
        C = L * L'
        P = cholinv(C)

        data = rand(rng, MvNormalMeanPrecision(m, P), n) |> eachcol |> collect .|> collect
        ## -------------------------------------------- ##
        ## Inference execution
        result    = inference_mv_wishart(data, n, d)
        result_km = inference_mv_wishart_known_mean(m, data, n, d)
        ## -------------------------------------------- ##
        ## Test inference results
        @test isapprox(mean(result.posteriors[:m]), m, atol = 0.05)
        @test isapprox(mean(result.posteriors[:P]), P, atol = 0.07)
        @test all(<(0), filter(e -> abs(e) > 1e-10, diff(result.free_energy)))

        @test isapprox(mean(result_km.posteriors[:P]), P, atol = 0.07)
        # Check that FE does not depend on iteration in the known mean cast
        @test all(==(first(result_km.free_energy)), result_km.free_energy)
        ## -------------------------------------------- ##
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd-mm-yyyy-HH-MM")
        plot_output      = joinpath(base_output, "mv_iid_wishart_$(timestamp)_v$(VERSION).png")
        benchmark_output = joinpath(base_output, "mv_iid_wishart_benchmark_$(timestamp)_v$(VERSION).txt")
        ## -------------------------------------------- ##
        ## Create output plots
        X = range(-5, 5, length = 200)
        Y = range(-5, 5, length = 200)

        p = plot(title = "MvIID experiment / Precision parametrisation")
        p = contour!(
            p,
            X,
            Y,
            (x, y) -> pdf(MvNormalMeanPrecision(mean(result.posteriors[:m]), mean(result.posteriors[:P])), [x, y]),
            label = "Estimated"
        )
        p = contour!(p, X, Y, (x, y) -> pdf(MvNormalMeanPrecision(m, P), [x, y]), label = "Real")
        savefig(p, plot_output)
        ## -------------------------------------------- ##
        ## Create output benchmarks (skip if CI)
        if get(ENV, "CI", nothing) != "true"
            benchmark = @benchmark inference_mv_wishart($data, $n, $d)#
            open(benchmark_output, "w") do io
                show(io, MIME("text/plain"), benchmark)
                versioninfo(io)
            end
        end
        ## -------------------------------------------- ##
    end

    @testset "Use case #2: Covariance parametrisation" begin
        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
        rng = StableRNG(123)

        n = 1500
        d = 2

        m = rand(rng, d)
        L = randn(rng, d, d)
        C = L * L'
        P = cholinv(C)

        data = rand(rng, MvNormalMeanCovariance(m, C), n) |> eachcol |> collect .|> collect
        ## -------------------------------------------- ##
        ## Inference execution
        result    = inference_mv_inverse_wishart(data, n, d)
        result_km = inference_mv_inverse_wishart_known_mean(m, data, n, d)
        ## -------------------------------------------- ##
        ## Test inference results
        @test isapprox(mean(result.posteriors[:m]), m, atol = 0.05)
        @test isapprox(mean(result.posteriors[:C]), C, atol = 0.15)
        @test all(<(0), filter(e -> abs(e) > 1e-10, diff(result.free_energy)))

        @test isapprox(mean(result_km.posteriors[:C]), C, atol = 0.15)
        # Check that FE does not depend on iteration in the known mean cast
        @test all(==(first(result_km.free_energy)), result_km.free_energy)
        ## -------------------------------------------- ##
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd-mm-yyyy-HH-MM")
        plot_output      = joinpath(base_output, "mv_iid_inverse_wishart_$(timestamp)_v$(VERSION).png")
        benchmark_output = joinpath(base_output, "mv_iid_inverse_wishart_benchmark_$(timestamp)_v$(VERSION).txt")
        ## -------------------------------------------- ##
        ## Create output plots
        X = range(-5, 5, length = 200)
        Y = range(-5, 5, length = 200)

        p = plot(title = "MvIID experiment / Covariance parametrisation")
        p = contour!(
            p,
            X,
            Y,
            (x, y) -> pdf(MvNormalMeanCovariance(mean(result.posteriors[:m]), mean(result.posteriors[:C])), [x, y]),
            label = "Estimated"
        )
        p = contour!(p, X, Y, (x, y) -> pdf(MvNormalMeanCovariance(m, C), [x, y]), label = "Real")
        savefig(p, plot_output)
        ## -------------------------------------------- ##
        ## Create output benchmarks (skip if CI)
        if get(ENV, "CI", nothing) != "true"
            benchmark = @benchmark inference_mv_inverse_wishart($data, $n, $d)#
            open(benchmark_output, "w") do io
                show(io, MIME("text/plain"), benchmark)
                versioninfo(io)
            end
        end
        ## -------------------------------------------- ##
    end
end

end
