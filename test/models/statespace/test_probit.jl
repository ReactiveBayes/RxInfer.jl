module RxInferModelsProbitTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

using StatsFuns: normcdf

# `include(test/utiltests.jl)`
include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

# Please use StableRNGs for random number generators

## Model definition
@model function probit_model(nr_samples::Int64)

    x = randomvar(nr_samples + 1)
    y = datavar(Float64, nr_samples)

    x[1] ~ Normal(mean = 0.0, precision = 0.01)

    for k in 2:nr_samples+1
        x[k] ~ Normal(mean = x[k-1] + 0.1, precision = 100)
        y[k-1] ~ Probit(x[k]) where {
            pipeline = RequireMessage(in = NormalMeanPrecision(0, 1.0))
        }
    end

end

## Inference definition
function probit_inference(data_y)
    return inference(
        model = probit_model(length(data_y)),
        data = (y = data_y, ),
        iterations = 10,
        returnvars = (x = KeepLast(), ),
        free_energy = true
    )
end

@testset "Probit Model" begin
    
    ## Data creation
    function generate_data(nr_samples::Int64; seed = 123)
        rng = StableRNG(seed)

        # hyper parameters
        u = 0.1

        # allocate space for data
        data_x = zeros(nr_samples + 1)
        data_y = zeros(nr_samples)

        # initialize data
        data_x[1] = -2

        # generate data
        for k in 2:nr_samples+1

            # calculate new x
            data_x[k] = data_x[k-1] + u + sqrt(0.01) * randn(rng)

            # calculate y
            data_y[k-1] = normcdf(data_x[k]) > rand(rng)
        end

        # return data
        return data_x, data_y
    end

    n = 40
    data_x, data_y = generate_data(n)
    
    ## Inference execution
    result = probit_inference(data_y)
    
    ## Test inference results
    @test length(result.free_energy) === 10
    @test last(result.free_energy) ≈ 15.646236967225065
    
    ## Create output plots
    @test_plot "models" "probit" begin
        mx = result.posteriors[:x]

        px = plot(xlabel = "t", ylabel = "x, y", legend = :bottomright)
        px = scatter!(px, data_y, label = "y")
        px = plot!(px, data_x[2:end], label = "x", lw = 2)
        px = plot!(px, mean.(mx)[2:end], ribbon = std.(mx)[2:end], fillalpha = 0.2, label = "x (inferred mean)")

        pf = plot(xlabel = "t", ylabel = "BFE")
        pf = plot!(pf, result.free_energy, label = "Bethe Free Energy")

        p = plot(px, pf, size = (800, 400))

        return p
    end

    @test_benchmark "models" "probit" probit_inference($data_y)
end

end
