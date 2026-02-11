@testitem "Probit Model" begin
    using BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs
    using StatsFuns: normcdf

    # `include(test/utiltests.jl)`
    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    # Please use StableRNGs for random number generators

    ## Model definition
    @model function probit_model(y, dependencies)
        x[1] ~ Normal(mean = 0.0, precision = 0.01)

        for k in 2:(length(y) + 1)
            x[k] ~ Normal(mean = x[k - 1] + 0.1, precision = 100)
            y[k - 1] ~ Probit(x[k]) where {dependencies = dependencies}
        end
    end

    ## Inference definition
    function probit_inference(data, dependencies)
        return infer(model = probit_model(dependencies = dependencies), data = (y = data,), iterations = 10, returnvars = KeepLast(), free_energy = true, callbacks = nothing)
    end

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
        for k in 2:(nr_samples + 1)

            # calculate new x
            data_x[k] = data_x[k - 1] + u + sqrt(0.01) * randn(rng)

            # calculate y
            data_y[k - 1] = normcdf(data_x[k]) > rand(rng)
        end

        # return data
        return data_x, data_y
    end

    n = 40
    data_x, data_y = generate_data(n)

    # `nothing` here should fallback to the `default` dependencies for the `Probit` node
    # Check that the result does not really depend on the initial value
    for dependencies in
        [nothing, RequireMessageFunctionalDependencies(in = NormalMeanPrecision(0.0, 1.0)), RequireMessageFunctionalDependencies(in = NormalMeanPrecision(0.0, 10.0))]
        result = probit_inference(data_y, dependencies)
        @test length(result.free_energy) === 10
        @test all(<=(1e-6), diff(result.free_energy)) # Some values are fluctuating due to approximations
        @test last(result.free_energy) ≈ 15.646236967225065
    end

    # We don't expect the `Probit` node to work properly when the `DefaultFunctionalDependencies` are being used
    @test_throws ErrorException probit_inference(data_y, DefaultFunctionalDependencies())

    result = probit_inference(data_y, RequireMessageFunctionalDependencies(in = NormalMeanPrecision(0.0, 1.0)))

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
