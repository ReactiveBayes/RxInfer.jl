@testitem "Probit Model" begin
    using BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs
    using StatsFuns: normcdf
    using ProbitMessagePassingRules

    # `include(test/utiltests.jl)`
    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    # Please use StableRNGs for random number generators

    ## Model definition
    @model function probit_model(y)
        x[1] ~ Normal(mean = 0.0, precision = 0.01)

        for k in 2:(length(y) + 1)
            x[k] ~ Normal(mean = x[k - 1] + 0.1, precision = 100)
            y[k - 1] ~ Probit(x[k])
        end
    end

    # The initial message on each Probit node's own `in` edge, in place of the node's
    @model function probit_model_with_initial_messages(y, init)
        x[1] ~ Normal(mean = 0.0, precision = 0.01)

        for k in 2:(length(y) + 1)
            x[k] ~ Normal(mean = x[k - 1] + 0.1, precision = 100)
            y[k - 1] ~ Probit(x[k]) where {initial_messages = (in = init,)}
        end
    end

    # v6's `where { dependencies = … }`, which the error below tells to replace
    @model function probit_model_with_dependencies(y, dependencies)
        x[1] ~ Normal(mean = 0.0, precision = 0.01)

        for k in 2:(length(y) + 1)
            x[k] ~ Normal(mean = x[k - 1] + 0.1, precision = 100)
            y[k - 1] ~ Probit(x[k]) where {dependencies = dependencies}
        end
    end

    ## Inference definition
    function probit_inference(data)
        return infer(
            model = probit_model(),
            data = (y = data,),
            iterations = 10,
            returnvars = KeepLast(),
            free_energy = true,
            disable_inference_error_hint = true,
        )
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

    # The `Probit` node declares the initial message on its `in` edge
    result = probit_inference(data_y)
    @test length(result.free_energy) === 10
    @test all(<=(1e-6), diff(result.free_energy)) # Some values are fluctuating due to approximations
    @test last(result.free_energy) ≈ 15.646236967225065

    # v6's `where { dependencies = RequireMessageFunctionalDependencies(in = init) }` gave this
    # free energy, iteration by iteration
    seeded = infer(
        model = probit_model_with_initial_messages(init = NormalMeanPrecision(0.0, 0.01)),
        data = (y = data_y,),
        iterations = 10,
        returnvars = KeepLast(),
        free_energy = true,
        disable_inference_error_hint = true,
    )
    @test seeded.free_energy ≈ [
        23.177871204365005, 15.743019214442128, 15.646693289905684, 15.646239480771953, 15.646237104462447,
        15.646236968964402, 15.646236967811586, 15.64623696724287, 15.646236967227736, 15.64623696722527,
    ]
    @test !(first(seeded.free_energy) ≈ first(result.free_energy))

    # A node's dependencies are declared by its algorithm now, and cannot be set in the model
    @test_throws "`where { initial_messages = (in = d,) }`" infer(
        model = probit_model_with_dependencies(dependencies = nothing),
        data = (y = data_y,),
        disable_inference_error_hint = true,
    )

    ## Create output plots
    @test_plot "models" "probit" begin
        mx = result.posteriors[:x]

        px = plot(xlabel = "t", ylabel = "x, y", legend = :bottomright)
        px = scatter!(px, data_y, label = "y")
        px = plot!(px, data_x[2:end], label = "x", lw = 2)
        px = plot!(
            px,
            mean.(mx)[2:end],
            ribbon = std.(mx)[2:end],
            fillalpha = 0.2,
            label = "x (inferred mean)",
        )

        pf = plot(xlabel = "t", ylabel = "BFE")
        pf = plot!(pf, result.free_energy, label = "Bethe Free Energy")

        p = plot(px, pf, size = (800, 400))

        return p
    end

    @test_benchmark "models" "probit" probit_inference($data_y)
end
