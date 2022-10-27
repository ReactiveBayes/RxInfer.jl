module ReactiveMPModelsNonLinearDynamicsTest

using Test, InteractiveUtils
using RxInfer, Distributions
using BenchmarkTools, Random, Plots, Dates, StableRNGs, Flux

# Please use StableRNGs for random number generators

# `include(test/utiltests.jl)`
include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

## Model definition
## -------------------------------------------- ##
sensor_location = 53
P = 5
sensor_var = 5
function f(z)
    (z - sensor_location)^2
end

@model function non_linear_dynamics(T, rng, n_iterations, n_samples, learning_rate)
    z = randomvar(T)
    x = randomvar(T)
    y = datavar(Float64, T)

    τ ~ GammaShapeRate(1.0, 1.0e-12)
    θ ~ GammaShapeRate(1.0, 1.0e-12)

    z[1] ~ NormalMeanPrecision(0, τ)
    x[1] ~ f(z[1]) where {meta = CVIApproximation(rng, n_iterations, n_samples, Descent(learning_rate))}
    y[1] ~ NormalMeanPrecision(x[1], θ)

    for t in 2:T
        z[t] ~ NormalMeanPrecision(z[t - 1] + 1, τ)
        x[t] ~ f(z[t]) where {meta = CVIApproximation(rng, n_iterations, n_samples, Descent(learning_rate))}
        y[t] ~ NormalMeanPrecision(x[t], θ)
    end

    return z, x, y
end

constraints = @constraints begin
    q(z, x, τ, θ) = q(z)q(x)q(τ)q(θ)
end

## -------------------------------------------- ##
## Inference definition
## -------------------------------------------- ##
function inference_cvi(transformed, rng, iterations)
    T = length(transformed)

    return inference(
        model = non_linear_dynamics(T, rng, 600, 600, 0.01),
        data = (y = transformed,),
        iterations = iterations,
        free_energy = true,
        returnvars = (z = KeepLast(),),
        constraints = constraints,
        initmessages = (z = NormalMeanVariance(0, P),),
        initmarginals = (z = NormalMeanVariance(0, P), τ = GammaShapeRate(1.0, 1.0e-12), θ = GammaShapeRate(1.0, 1.0e-12))
    )
end

@testset "Non linear dynamics" begin
    @testset "Use case #1" begin
        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
        seed = 123

        rng = MersenneTwister(seed)

        # For large `n` apply: smoothing(model_options(limit_stack_depth = 500), ...)
        T = 50

        sensor_location = 53

        hidden = collect(1:T)
        data = (hidden + rand(rng, NormalMeanVariance(0.0, sqrt(P)), T))
        transformed = (data .- sensor_location) .^ 2 + rand(rng, NormalMeanVariance(0.0, sensor_var), T)
        ## -------------------------------------------- ##
        ## Inference execution
        res = inference_cvi(transformed, rng, 110)
        ## -------------------------------------------- ##
        ## Test inference results should be there
        ## -------------------------------------------- ##
        mz = res.posteriors[:z]
        fe = res.free_energy
        @test length(res.posteriors[:z]) === T
        @test all(mean.(mz) .- 6 .* std.(mz) .< hidden .< (mean.(mz) .+ 6 .* std.(mz)))
        @test (sum((mean.(mz) .- 4 .* std.(mz)) .< hidden .< (mean.(mz) .+ 4 .* std.(mz))) / T) > 0.95
        @test (sum((mean.(mz) .- 3 .* std.(mz)) .< hidden .< (mean.(mz) .+ 3 .* std.(mz))) / T) > 0.90
        @test abs(last(fe) - 362.6552215247382) < 0.01

        @test (first(fe) - last(fe)) > 0
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd-mm-yyyy-HH-MM")
        plot_output      = joinpath(base_output, "cvi_plot_$(timestamp)_v$(VERSION).png")
        benchmark_output = joinpath(base_output, "non_linear_dynamics_$(timestamp)_v$(VERSION).txt")

        ## Create output plots
        @test_plot "models" "hgf" begin
            px = plot()

            px = plot!(px, hidden, label = "Hidden Signal", color = :red)
            px = plot!(px, map(mean, res.posteriors[:z]), label = "Estimated signal location", color = :orange)
            px = plot!(px, map(mean, res.posteriors[:z]), ribbon = (9 .* var.(res.posteriors[:z])) .|> sqrt, fillalpha = 0.5, label = "Estimated Signal confidence", color = :blue)
            pf = plot(fe, label = "Bethe Free Energy")

            return plot(px, pf, layout = @layout([a; b]))
        end

        @test_benchmark "models" "cvi" inference_cvi($transformed, $rng, 110)
    end
end

end
