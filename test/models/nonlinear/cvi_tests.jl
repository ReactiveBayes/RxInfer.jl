@testitem "Non linear dynamics with the CVI algorithm" begin
    using Distributions
    using BenchmarkTools, Plots, StableRNGs, Optimisers, Random, Dates

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

    @model function non_linear_dynamics(y)
        τ ~ Gamma(shape = 0.01, rate = 0.01)
        θ ~ Gamma(shape = 0.01, rate = 0.01)

        z[1] ~ Normal(mean = 0, precision = τ)
        x[1] := f(z[1])
        y[1] ~ Normal(mean = x[1], precision = θ)

        for t in 2:length(y)
            z[t] ~ Normal(mean = z[t - 1] + 1, precision = τ)
            x[t] := f(z[t])
            y[t] ~ Normal(mean = x[t], precision = θ)
        end
    end

    constraints = @constraints begin
        q(z, x, τ, θ) = q(z)q(x)q(τ)q(θ)
    end

    @meta function model_meta(rng, n_iterations, n_samples, learning_rate)
        f() -> CVI(rng, n_iterations, n_samples, Optimisers.Descent(learning_rate))
    end

    init = @initialization begin
        μ(z) = NormalMeanVariance(0, P)
        q(z) = NormalMeanVariance(0, P)
        q(τ) = GammaShapeRate(1e-12, 1e-3)
        q(θ) = GammaShapeRate(1e-12, 1e-3)
    end

    function inference_cvi(transformed, rng, iterations)
        return infer(
            model = non_linear_dynamics(),
            data = (y = transformed,),
            iterations = iterations,
            free_energy = true,
            returnvars = (z = KeepLast(),),
            constraints = constraints,
            meta = model_meta(rng, 600, 600, 0.01),
            initialization = init
        )
    end

    seed = 42
    rng = StableRNG(seed)
    T = 50

    sensor_location = 53

    hidden = collect(1:T)
    data = (hidden + rand(rng, NormalMeanVariance(0.0, sqrt(P)), T))
    transformed = (data .- sensor_location) .^ 2 + rand(rng, NormalMeanVariance(0.0, sensor_var), T)
    ## -------------------------------------------- ##
    ## Inference execution
    res = inference_cvi(transformed, rng, 150)
    ## -------------------------------------------- ##
    mz = res.posteriors[:z]
    fe = res.free_energy

    @test length(res.posteriors[:z]) === T

    @test all(mean.(mz) .- 6 .* std.(mz) .< hidden .< (mean.(mz) .+ 6 .* std.(mz)))
    @test (sum((mean.(mz) .- 4 .* std.(mz)) .< hidden .< (mean.(mz) .+ 4 .* std.(mz))) / T) > 0.95
    @test (sum((mean.(mz) .- 3 .* std.(mz)) .< hidden .< (mean.(mz) .+ 3 .* std.(mz))) / T) > 0.90

    # Free energy for the CVI may fluctuate
    @test all(d -> d < 3.0, diff(fe)) # Check that the fluctuations are not big
    @test abs(last(fe) - 317) < 1.0   # Check the final result with relatively low precision
    @test (first(fe) - last(fe)) > 0

    ## Create output plots
    @test_plot "models" "cvi" begin
        px = plot()

        px = plot!(px, hidden, label = "Hidden Signal", color = :red)
        px = plot!(px, map(mean, res.posteriors[:z]), label = "Estimated signal location", color = :orange)
        px = plot!(px, map(mean, res.posteriors[:z]), ribbon = (9 .* var.(res.posteriors[:z])) .|> sqrt, fillalpha = 0.5, label = "Estimated Signal confidence", color = :blue)
        pf = plot(fe, label = "Bethe Free Energy")

        return plot(px, pf, layout = @layout([a; b]))
    end

    @test_benchmark "models" "cvi" inference_cvi($transformed, $rng, 110)
end
