@testitem "Model mixture" begin
    using Distributions
    using BenchmarkTools, LinearAlgebra, StableRNGs, Plots

    # Please use StableRNGs for random number generators

    # `include(test/utiltests.jl)`
    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    ## Model definition
    ## -------------------------------------------- ##

    @model function beta_model1(y)
        θ ~ Beta(4.0, 8.0)
        for i in eachindex(y)
            y[i] ~ Bernoulli(θ)
        end
    end

    @model function beta_model2(y)
        θ ~ Beta(8.0, 4.0)
        for i in eachindex(y)
            y[i] ~ Bernoulli(θ)
        end
    end

    @model function beta_mixture_model(y)
        selector ~ Bernoulli(0.7)
        in1 ~ Beta(4.0, 8.0)
        in2 ~ Beta(8.0, 4.0)
        θ ~ Mixture(switch = selector, inputs = [in1, in2])
        for i in eachindex(y)
            y[i] ~ Bernoulli(θ)
        end
    end
    @testset "Check inference results" begin
        ## -------------------------------------------- ##
        ## Data creation
        ## -------------------------------------------- ##
        rng = StableRNG(42)
        n = 20
        θ_real = 0.75
        distribution = Bernoulli(θ_real)
        dataset = float.(rand(rng, Bernoulli(θ_real), n))

        ## -------------------------------------------- ##
        ## Inference execution
        result1 = infer(model = beta_model1(), data = (y = dataset,), returnvars = (θ = KeepLast(),), free_energy = true, addons = AddonLogScale())
        result2 = infer(model = beta_model2(), data = (y = dataset,), returnvars = (θ = KeepLast(),), free_energy = true, addons = AddonLogScale())

        resultswitch = infer(
            model = beta_mixture_model(), data = (y = dataset,), returnvars = (θ = KeepLast(), in1 = KeepLast(), in2 = KeepLast(), selector = KeepLast()), addons = AddonLogScale()
        )

        ## -------------------------------------------- ##
        ## Test inference results

        # check inference results
        @test getdata(result1.posteriors[:θ]) == getdata(resultswitch.posteriors[:in1])
        @test getdata(result2.posteriors[:θ]) == getdata(resultswitch.posteriors[:in2])
        @test getdata(resultswitch.posteriors[:in1]) == component(getdata(resultswitch.posteriors[:θ]), 1)
        @test getdata(resultswitch.posteriors[:in2]) == component(getdata(resultswitch.posteriors[:θ]), 2)
        @test getdata(resultswitch.posteriors[:selector]).p ≈ getdata(resultswitch.posteriors[:θ]).weights

        # check free energies
        @test -result1.free_energy[1] ≈ getlogscale(result1.posteriors[:θ])
        @test -result2.free_energy[1] ≈ getlogscale(result2.posteriors[:θ])
        @test getlogscale(resultswitch.posteriors[:in1]) ≈ log(0.3) - result1.free_energy[1]
        @test getlogscale(resultswitch.posteriors[:in2]) ≈ log(0.7) - result2.free_energy[1]
        @test log(0.3 * exp(-result1.free_energy[1]) + 0.7 * exp(-result2.free_energy[1])) ≈ getlogscale(resultswitch.posteriors[:selector])
        @test log(0.3 * exp(-result1.free_energy[1]) + 0.7 * exp(-result2.free_energy[1])) ≈ getlogscale(resultswitch.posteriors[:θ])
        @test getlogscale(resultswitch.posteriors[:θ]) ≈ getlogscale(resultswitch.posteriors[:selector])

        ## Create output plots
        @test_plot "models" "switch" begin
            rθ = range(0, 1, length = 1000)
            θestimated = resultswitch.posteriors[:θ]
            p = plot(title = "Inference results")

            plot!(rθ, (x) -> pdf(MixtureDistribution([Beta(4.0, 8.0), Beta(8.0, 4.0)], [0.5, 0.5]), x), fillalpha = 0.3, fillrange = 0, label = "P(θ)", c = 1)
            plot!(rθ, (x) -> pdf(getdata(θestimated), x), fillalpha = 0.3, fillrange = 0, label = "P(θ|y)", c = 3)
            vline!([θ_real], label = "Real θ")
            return p
        end
    end
end
