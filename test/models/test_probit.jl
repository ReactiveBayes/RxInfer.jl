module RxInferModelsProbitTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

using StatsFuns: normcdf

# Please use StableRNGs for random number generators

## Model definition
## -------------------------------------------- ##
@model function probit_model(nr_samples::Int64)

    # allocate space for variables
    x = randomvar(nr_samples + 1)
    y = datavar(Float64, nr_samples)

    # specify uninformative prior
    x[1] ~ NormalMeanPrecision(0.0, 0.01)

    # create model 
    for k in 2:nr_samples+1
        x[k] ~ NormalMeanPrecision(x[k-1] + 0.1, 100)
        y[k-1] ~ Probit(x[k]) where {
            pipeline = RequireMessage(in = NormalMeanPrecision(0, 1.0))
        }
    end

    # return parameters
    return x, y
end

## -------------------------------------------- ##
## Inference definition
## -------------------------------------------- ##
function probit_inference(data_y)
    return inference(
        model = Model(probit_model, length(data_y)),
        data = (
            y = data_y,
        ),
        iterations = 10,
        returnvars = (
            x = KeepLast(),
        ),
        free_energy = true
    )
end

@testset "Probit Model" begin
    @testset "Use case #1" begin
        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
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
        ## -------------------------------------------- ##
        ## Inference execution
        result = probit_inference(data_y)
        ## -------------------------------------------- ##
        ## Test inference results
        @test length(result.free_energy) === 10
        @test last(result.free_energy) ≈ 15.646236967225065
        ## -------------------------------------------- ##
        ## Form debug output
        base_output = joinpath(pwd(), "_output", "models")
        mkpath(base_output)
        timestamp        = Dates.format(now(), "dd-mm-yyyy-HH-MM")
        plot_output      = joinpath(base_output, "probit_model_plot_$(timestamp)_v$(VERSION).png")
        benchmark_output = joinpath(base_output, "probit_model_benchmark_$(timestamp)_v$(VERSION).txt")
        ## -------------------------------------------- ##
        ## Create output plots
        mx = result.posteriors[:x]

        px = plot(xlabel = "t", ylabel = "x, y", legend = :bottomright)
        px = scatter!(px, data_y, label = "y")
        px = plot!(px, data_x[2:end], label = "x", lw = 2)
        px = plot!(px, mean.(mx)[2:end], ribbon = std.(mx)[2:end], fillalpha = 0.2, label = "x (inferred mean)")

        pf = plot(xlabel = "t", ylabel = "BFE")
        pf = plot!(pf, result.free_energy, label = "Bethe Free Energy")

        p = plot(px, pf, size = (800, 400))
        savefig(p, plot_output)
        ## -------------------------------------------- ##
        ## Create output benchmarks (skip if CI)
        if get(ENV, "CI", nothing) != "true"
            benchmark = @benchmark probit_inference($data_y) seconds = 15#
            open(benchmark_output, "w") do io
                show(io, MIME("text/plain"), benchmark)
                versioninfo(io)
            end
        end
        ## -------------------------------------------- ##
    end
end

end
