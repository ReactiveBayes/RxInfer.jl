@testitem "Static inputs in nonlinear deterministic factor nodes" begin
    using Test, InteractiveUtils
    using RxInfer, BenchmarkTools, Random, Plots, LinearAlgebra, StableRNGs

    include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

    ## Model definition
    ## -------------------------------------------- ##

    function f₂(x, θ)
        return x .+ θ
    end

    @model function delta_2inputs_θ_fixed(meta, θ, y)
        c = zeros(2)
        c[1] = 1.0

        x ~ MvNormal(μ = zeros(2), Λ = diageye(2))
        z := f₂(x, θ) where {meta = meta}
        w ~ Normal(μ = dot(z, c), σ² = 1.0)
        y ~ Normal(μ = w, σ² = 0.5)
    end

    @model function delta_2inputs_x_fixed(meta, x, y)
        c = zeros(2)
        c[1] = 1.0

        θ ~ MvNormal(μ = ones(2), Λ = diageye(2))
        z := f₂(x, θ) where {meta = meta}
        w ~ Normal(μ = dot(z, c), σ² = 1.0)
        y ~ Normal(μ = w, σ² = 0.5)
    end

    ## -------------------------------------------- ##
    ## Inference definition
    ## -------------------------------------------- ##

    function inference_2inputs_θ_fixed(data, θ)
        metas = (DeltaMeta(method = Linearization()), DeltaMeta(method = Unscented()), Linearization(), Unscented())

        datavar_based = map(metas) do meta
            return infer(model = delta_2inputs_θ_fixed(meta = meta), data = (y = data, θ = θ), free_energy = true)
        end

        constvar_based = map(metas) do meta
            return infer(model = delta_2inputs_θ_fixed(meta = meta, θ = θ), data = (y = data,), free_energy = true)
        end

        foreach(zip(datavar_based, constvar_based)) do (d, c)
            @test !haskey(d.posteriors, :θ)
            @test !haskey(c.posteriors, :θ)
            @test mean(d.posteriors[:z]) ≈ mean(c.posteriors[:z])
            @test mean(d.posteriors[:x]) ≈ mean(c.posteriors[:x])
            @test d.free_energy == c.free_energy
        end
    end

    function inference_2inputs_x_fixed(data, x)
        metas = (DeltaMeta(method = Linearization()), DeltaMeta(method = Unscented()), Linearization(), Unscented())

        datavar_based = map(metas) do meta
            return infer(model = delta_2inputs_x_fixed(meta = meta), data = (y = data, x = x), free_energy = true)
        end

        constvar_based = map(metas) do meta
            return infer(model = delta_2inputs_x_fixed(meta = meta, x = x), data = (y = data,), free_energy = true)
        end

        foreach(zip(datavar_based, constvar_based)) do (d, c)
            @test !haskey(d.posteriors, :x)
            @test !haskey(c.posteriors, :x)
            @test mean(d.posteriors[:z]) ≈ mean(c.posteriors[:z])
            @test mean(d.posteriors[:θ]) ≈ mean(c.posteriors[:θ])
            @test d.free_energy == c.free_energy
        end
    end

    # Inference execution
    inference_2inputs_θ_fixed(4.0, [1.0, 2.0])
    inference_2inputs_x_fixed(4.0, [1.0, 2.0])

    # All models have been created and runned extra tests inside. 
    # The inference finished without errors
    @test true
end
