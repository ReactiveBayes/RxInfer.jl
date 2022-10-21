module RxInferNonlinearityModelsDeltaTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, LinearAlgebra, StableRNGs

# `include(test/utiltests.jl)`
include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

# Please use StableRNGs for random number generators

## Model definition
## -------------------------------------------- ##

function f₁(x)
    return sqrt.(x)
end

function f₁_inv(x)
    return x .^ 2
end

@model function delta_1input(meta)
    y2 = datavar(Float64)
    c = zeros(2)
    c[1] = 1.0

    x ~ MvNormal(μ = ones(2), Λ = diageye(2))
    z ~ f₁(x) where {meta = meta}
    y1 ~ Normal(μ = dot(z, c), σ² = 1.0)
    y2 ~ Normal(μ = y1, σ² = 0.5)
end

function f₂(x, θ)
    return x .+ θ
end

function f₂_x(θ, z)
    return z .- θ
end

function f₂_θ(x, z)
    return z .- x
end

@model function delta_2inputs(meta)
    y2 = datavar(Float64)
    c = zeros(2)
    c[1] = 1.0

    θ ~ MvNormal(μ = ones(2), Λ = diageye(2))
    x ~ MvNormal(μ = zeros(2), Λ = diageye(2))
    z ~ f₂(x, θ) where {meta = meta}
    y1 ~ Normal(μ = dot(z, c), σ² = 1.0)
    y2 ~ Normal(μ = y1, σ² = 0.5)
end

function f₃(x, θ, ζ)
    return x .+ θ .+ ζ
end

@model function delta_3inputs(meta)
    y2 = datavar(Float64)
    c = zeros(2)
    c[1] = 1.0

    θ ~ MvNormal(μ = ones(2), Λ = diageye(2))
    ζ ~ MvNormal(μ = 0.5ones(2), Λ = diageye(2))
    x ~ MvNormal(μ = zeros(2), Λ = diageye(2))
    z ~ f₃(x, θ, ζ) where {meta = meta}
    y1 ~ Normal(μ = dot(z, c), σ² = 1.0)
    y2 ~ Normal(μ = y1, σ² = 0.5)
end

function f₄(x, θ)
    return θ .* x
end

@model function delta_2input_1d2d(meta)
    y2 = datavar(Float64)
    c = zeros(2)
    c[1] = 1.0

    θ ~ Normal(μ = 0.5, γ = 1.0)
    x ~ MvNormal(μ = zeros(2), Λ = diageye(2))
    z ~ f₄(x, θ) where {meta = meta}
    y1 ~ Normal(μ = dot(z, c), σ² = 1.0)
    y2 ~ Normal(μ = y1, σ² = 0.5)
end

## -------------------------------------------- ##
## Inference definition
## -------------------------------------------- ##
function inference_1input(data)

    # We test here different approximation methods
    metas = (
        DeltaMeta(method = Linearization(), inverse = f₁_inv),
        DeltaMeta(method = Unscented(), inverse = f₁_inv),
        DeltaMeta(method = Linearization()),
        DeltaMeta(method = Unscented()),
        Linearization(),
        Unscented()
    )

    return map(metas) do meta
        return inference(model = delta_1input(meta), data = (y2 = data,), free_energy = true, free_energy_diagnostics = (BetheFreeEnergyCheckNaNs(), BetheFreeEnergyCheckInfs()))
    end
end

function inference_2inputs(data)
    metas = (
        DeltaMeta(method = Linearization(), inverse = (f₂_x, f₂_θ)),
        DeltaMeta(method = Unscented(), inverse = (f₂_x, f₂_θ)),
        DeltaMeta(method = Linearization()),
        DeltaMeta(method = Unscented()),
        Linearization(),
        Unscented()
    )

    return map(metas) do meta
        return inference(model = delta_2inputs(meta), data = (y2 = data,), free_energy = true, free_energy_diagnostics = (BetheFreeEnergyCheckNaNs(), BetheFreeEnergyCheckInfs()))
    end
end

function inference_3inputs(data)
    metas = (DeltaMeta(method = Linearization()), DeltaMeta(method = Unscented()), Linearization(), Unscented())

    return map(metas) do meta
        return inference(model = delta_3inputs(meta), data = (y2 = data,), free_energy = true, free_energy_diagnostics = (BetheFreeEnergyCheckNaNs(), BetheFreeEnergyCheckInfs()))
    end
end

function inference_2input_1d2d(data)
    metas = (DeltaMeta(method = Linearization()), DeltaMeta(method = Unscented()), Linearization(), Unscented())

    return map(metas) do meta
        return inference(
            model = delta_2input_1d2d(meta), data = (y2 = data,), free_energy = true, free_energy_diagnostics = (BetheFreeEnergyCheckNaNs(), BetheFreeEnergyCheckInfs())
        )
    end
end

@testset "Nonlinear models: generic applicability" begin
    @testset "Linearization, Unscented transforms" begin
        ## -------------------------------------------- ##
        ## Data creation
        data = 4.0
        ## -------------------------------------------- ##
        ## Inference execution
        result₁ = inference_1input(data)
        result₂ = inference_2inputs(data)
        result₃ = inference_3inputs(data)
        result₄ = inference_2input_1d2d(data)

        ## All models have been created. The inference finished without errors ##
        @test true
    end
end

end
