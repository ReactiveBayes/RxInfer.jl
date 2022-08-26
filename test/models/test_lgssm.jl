module RxInferModelsLGSSMTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

## Model definition
## -------------------------------------------- ##
@model function univariate_lgssm_model(n, x0, c_, P_)
    x_prior ~ NormalMeanVariance(mean(x0), cov(x0))

    x = randomvar(n)
    c = constvar(c_)
    P = constvar(P_)
    y = datavar(Float64, n)

    x_prev = x_prior

    for i in 1:n
        x[i] ~ x_prev + c
        y[i] ~ NormalMeanVariance(x[i], P)
        x_prev = x[i]
    end

    return x, y
end
## -------------------------------------------- ##
@model function multivariate_lgssm_model(n, x0, A, B, Q, P)

    # We create constvar references for better efficiency
    cA = constvar(A)
    cB = constvar(B)
    cQ = constvar(Q)
    cP = constvar(P)

    # `x` is a sequence of hidden states
    x = randomvar(n)
    # `y` is a sequence of "clamped" observations
    y = datavar(Vector{Float64}, n)

    x_prior ~ MvNormalMeanCovariance(mean(x0), cov(x0))
    x_prev = x_prior

    for i in 1:n
        x[i] ~ MvNormalMeanCovariance(cA * x_prev, cQ)
        y[i] ~ MvNormalMeanCovariance(cB * x[i], cP)
        x_prev = x[i]
    end

    return x, y
end
## -------------------------------------------- ##
## Inference definition
## -------------------------------------------- ##
function univariate_lgssm_inference(data, x0, c, P)
    n = length(data)

    model, (x, y) = univariate_lgssm_model(n, x0, c, P)

    x_buffer = buffer(Marginal, n)
    fe       = keep(Float64)

    x_sub = subscribe!(getmarginals(x), x_buffer)
    f_sub = subscribe!(score(Float64, BetheFreeEnergy(), model), fe)

    update!(y, data)

    unsubscribe!((x_sub, f_sub))

    return x_buffer, fe
end
## -------------------------------------------- ##
function multivariate_lgssm_inference(data, x0, A, B, Q, P)
    n = length(data)

    model, (x, y) = multivariate_lgssm_model(model_options(limit_stack_depth = 500), n, x0, A, B, Q, P)

    xbuffer = buffer(Marginal, n)
    fe      = keep(Float64)

    x_sub = subscribe!(getmarginals(x), xbuffer)
    f_sub = subscribe!(score(Float64, BetheFreeEnergy(), model), fe)

    update!(y, data)

    unsubscribe!((x_sub, f_sub))

    return xbuffer, fe
end
## -------------------------------------------- ##

@testset "Linear Gaussian State Space Model" begin
    @testset "Univariate" begin
        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
        rng      = StableRNG(123)
        P        = 100.0
        n        = 500
        hidden   = collect(1:n)
        data     = hidden + rand(rng, Normal(0.0, sqrt(P)), n)
        x0_prior = NormalMeanVariance(0.0, 10000.0)
        ## -------------------------------------------- ##
        ## Inference execution
        x_estimated, fe = univariate_lgssm_inference(data, x0_prior, 1.0, P)
        ## -------------------------------------------- ##
        ## Test inference results
        @test length(x_estimated) === n
        @test all(
            (mean.(x_estimated) .- 3 .* std.(x_estimated)) .< hidden .< (mean.(x_estimated) .+ 3 .* std.(x_estimated))
        )
        @test all(var.(x_estimated) .> 0.0)
        @test length(fe) === 1
        @test abs(last(fe) - 1854.297647) < 0.01
        ## -------------------------------------------- ##
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd-mm-yyyy-HH-MM")
        plot_output      = joinpath(base_output, "lgssm_univariate_plot_$(timestamp)_v$(VERSION).png")
        benchmark_output = joinpath(base_output, "lgssm_univariate_benchmark_$(timestamp)_v$(VERSION).txt")
        ## -------------------------------------------- ##
        ## Create output plots
        subrange = 200:215
        m = mean.(x_estimated)[subrange]
        s = std.(x_estimated)[subrange]
        p = plot()
        p = plot!(subrange, m, ribbon = s, label = "Estimated signal")
        p = plot!(subrange, hidden[subrange], label = "Hidden signal")
        p = scatter!(subrange, data[subrange], label = "Observations")
        savefig(p, plot_output)
        ## -------------------------------------------- ##
        ## Create output benchmarks (skip if CI)
        if get(ENV, "CI", nothing) != "true"
            benchmark = @benchmark univariate_lgssm_inference($data, $x0_prior, 1.0, $P) seconds = 15
            open(benchmark_output, "w") do io
                show(io, MIME("text/plain"), benchmark)
                versioninfo(io)
            end
        end
        ## -------------------------------------------- ##
    end

    @testset "Multivariate" begin
        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
        function generate_data(rng, A, B, Q, P)
            x_prev = [10.0, -10.0]

            x = Vector{Vector{Float64}}(undef, n)
            y = Vector{Vector{Float64}}(undef, n)

            for i in 1:n
                x[i] = rand(rng, MvNormal(A * x_prev, Q))
                y[i] = rand(rng, MvNormal(B * x[i], P))
                x_prev = x[i]
            end

            return x, y
        end
        ## -------------------------------------------- ##
        # Seed for reproducibility
        rng = StableRNG(1234)
        # We will model 2-dimensional observations with rotation matrix `A`
        # To avoid clutter we also assume that matrices `A`, `B`, `P` and `Q`
        # are known and fixed for all time-steps
        θ = π / 35
        A = [cos(θ) -sin(θ); sin(θ) cos(θ)]
        B = diageye(2)
        Q = diageye(2)
        P = 25.0 .* diageye(2)
        n = 1_000 # Number of observations
        x, y = generate_data(rng, A, B, Q, P)
        x0 = MvNormalMeanCovariance(zeros(2), 100.0 * diageye(2))
        ## -------------------------------------------- ##
        ## Inference execution
        xmarginals, fe = multivariate_lgssm_inference(y, x0, A, B, Q, P)
        ## Test inference results
        @test length(xmarginals) === n
        # We use 3.0var instead of 3.0std here for easier dot broadcasting with mean
        @test all((mean.(xmarginals) .- 3.0 .* var.(xmarginals)) .< x .< (mean.(xmarginals) .+ 3.0 .* var.(xmarginals)))
        @test all(isposdef.(cov.(xmarginals)))
        @test length(fe) === 1
        @test abs(last(fe) - 6275.9015944677) < 0.01
        ## -------------------------------------------- ##
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd-mm-yyyy-HH-MM")
        plot_output      = joinpath(base_output, "lgssm_multivariate_plot_$(timestamp)_v$(VERSION).png")
        benchmark_output = joinpath(base_output, "lgssm_multivariate_benchmark_$(timestamp)_v$(VERSION).txt")
        ## -------------------------------------------- ##
        ## Create output plots
        slicedim(dim) = (a) -> map(e -> e[dim], a)

        subrange = 100:500
        px = plot()

        px = plot!(px, x[subrange] |> slicedim(1), label = "Hidden Signal (dim-1)", color = :orange)
        px = plot!(px, x[subrange] |> slicedim(2), label = "Hidden Signal (dim-2)", color = :green)

        px = plot!(
            px,
            mean.(xmarginals)[subrange] |> slicedim(1),
            ribbon = var.(xmarginals)[subrange] |> slicedim(1) .|> sqrt,
            fillalpha = 0.5,
            label = "Estimated Signal (dim-1)",
            color = :teal
        )
        px = plot!(
            px,
            mean.(xmarginals)[subrange] |> slicedim(2),
            ribbon = var.(xmarginals)[subrange] |> slicedim(2) .|> sqrt,
            fillalpha = 0.5,
            label = "Estimated Signal (dim-1)",
            color = :violet
        )

        savefig(px, plot_output)
        ## -------------------------------------------- ##
        ## Create output benchmarks (skip if CI)
        if get(ENV, "CI", nothing) != "true"
            benchmark = @benchmark multivariate_lgssm_inference($y, $x0, $A, $B, $Q, $P) seconds = 15
            open(benchmark_output, "w") do io
                show(io, MIME("text/plain"), benchmark)
                versioninfo(io)
            end
        end
        ## -------------------------------------------- ##
    end
end

end
