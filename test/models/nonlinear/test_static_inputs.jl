module RxInferNonlinearityModelsDeltaTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, LinearAlgebra, StableRNGs

# `include(test/utiltests.jl)`
include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

# Please use StableRNGs for random number generators

## Model definition
## -------------------------------------------- ##

function f₂(x, θ)
    return x .+ θ
end

@model function delta_2inputs_θ_datavar_fixed(meta)
    y2 = datavar(Float64)
    c = zeros(2)
    c[1] = 1.0

    θ = datavar(Vector{Float64})
    x ~ MvNormal(μ = zeros(2), Λ = diageye(2))
    z ~ f₂(x, θ) where {meta = meta}
    y1 ~ Normal(μ = dot(z, c), σ² = 1.0)
    y2 ~ Normal(μ = y1, σ² = 0.5)
end

@model function delta_2inputs_θ_constvar_fixed(meta, θ)
    y2 = datavar(Float64)
    c = zeros(2)
    c[1] = 1.0

    x ~ MvNormal(μ = zeros(2), Λ = diageye(2))
    z ~ f₂(x, θ) where {meta = meta}
    y1 ~ Normal(μ = dot(z, c), σ² = 1.0)
    y2 ~ Normal(μ = y1, σ² = 0.5)
end

@model function delta_2inputs_x_datavar_fixed(meta)
    y2 = datavar(Float64)
    c = zeros(2)
    c[1] = 1.0

    θ ~ MvNormal(μ = ones(2), Λ = diageye(2))
    x = datavar(Vector{Float64})
    z ~ f₂(x, θ) where {meta = meta}
    y1 ~ Normal(μ = dot(z, c), σ² = 1.0)
    y2 ~ Normal(μ = y1, σ² = 0.5)
end

@model function delta_2inputs_x_constvar_fixed(meta, x)
    y2 = datavar(Float64)
    c = zeros(2)
    c[1] = 1.0

    θ ~ MvNormal(μ = ones(2), Λ = diageye(2))
    z ~ f₂(x, θ) where {meta = meta}
    y1 ~ Normal(μ = dot(z, c), σ² = 1.0)
    y2 ~ Normal(μ = y1, σ² = 0.5)
end

## -------------------------------------------- ##
## Inference definition
## -------------------------------------------- ##

function inference_2inputs_θ_fixed(data, θ)
    metas = (DeltaMeta(method = Linearization()), DeltaMeta(method = Unscented()), Linearization(), Unscented())

    datavar_based = map(metas) do meta
        return inference(model = delta_2inputs_θ_datavar_fixed(meta), data = (y2 = data, θ = θ), free_energy = true)
    end

    constvar_based = map(metas) do meta
        return inference(model = delta_2inputs_θ_constvar_fixed(meta, θ), data = (y2 = data,), free_energy = true)
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
        return inference(model = delta_2inputs_x_datavar_fixed(meta), data = (y2 = data, x = x), free_energy = true)
    end

    constvar_based = map(metas) do meta
        return inference(model = delta_2inputs_x_constvar_fixed(meta, x), data = (y2 = data,), free_energy = true)
    end

    foreach(zip(datavar_based, constvar_based)) do (d, c)
        @test !haskey(d.posteriors, :x)
        @test !haskey(c.posteriors, :x)
        @test mean(d.posteriors[:z]) ≈ mean(c.posteriors[:z])
        @test mean(d.posteriors[:θ]) ≈ mean(c.posteriors[:θ])
        @test d.free_energy == c.free_energy
    end
end

@testset "Nonlinear models: static inputs" begin
    ## -------------------------------------------- ##
    ## Inference execution
    inference_2inputs_θ_fixed(4.0, [1.0, 2.0])
    inference_2inputs_x_fixed(4.0, [1.0, 2.0])

    ## All models have been created. The inference finished without errors ##
    @test true
end

end
