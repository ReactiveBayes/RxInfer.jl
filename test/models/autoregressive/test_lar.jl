module RxInferModelsAutoregressiveTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, LinearAlgebra, StableRNGs

# `include(test/utiltests.jl)`
include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

@model function lar_model(::Type{Multivariate}, n, order, c, stype, τ)

    # Parameter priors
    γ ~ Gamma(shape = 1.0, rate = 1.0)
    θ ~ MvNormal(mean = zeros(order), precision = diageye(order))

    # We create a sequence of random variables for hidden states
    x = randomvar(n)
    # As well a sequence of observartions
    y = datavar(Float64, n)

    ct = constvar(c)
    # We assume observation noise to be known
    cτ = constvar(τ)

    # Prior for first state
    x0 ~ MvNormal(mean = zeros(order), precision = diageye(order))

    x_prev = x0

    # AR process requires extra meta information
    meta = ARMeta(Multivariate, order, stype)

    for i in 1:n
        # Autoregressive node uses structured factorisation assumption between states
        x[i] ~ AR(x_prev, θ, γ) where {q = q(y, x)q(γ)q(θ), meta = meta}
        y[i] ~ Normal(mean = dot(ct, x[i]), precision = cτ)
        x_prev = x[i]
    end
end

@model function lar_model(::Type{Univariate}, n, order, c, stype, τ)

    # Parameter priors
    γ ~ Gamma(shape = 1.0, rate = 1.0)
    θ ~ Normal(mean = 0.0, precision = 1.0)

    # We create a sequence of random variables for hidden states
    x = randomvar(n)
    # As well a sequence of observartions
    y = datavar(Float64, n)

    ct = constvar(c)
    # We assume observation noise to be known
    cτ = constvar(τ)

    # Prior for first state
    x0 ~ Normal(mean = 0.0, precision = 1.0)

    x_prev = x0

    # AR process requires extra meta information
    meta = ARMeta(Univariate, order, stype)

    for i in 1:n
        x[i] ~ AR(x_prev, θ, γ) where {q = q(y, x)q(γ)q(θ), meta = meta}
        y[i] ~ Normal(mean = ct * x[i], precision = cτ)
        x_prev = x[i]
    end
end

function lar_init_marginals(::Type{Multivariate}, order)
    return (γ = GammaShapeRate(1.0, 1.0), θ = MvNormalMeanPrecision(zeros(order), diageye(order)))
end

function lar_init_marginals(::Type{Univariate}, order)
    return (γ = GammaShapeRate(1.0, 1.0), θ = NormalMeanPrecision(0.0, 1.0))
end

function lar_inference(data, order, artype, stype, niter, τ)
    n = length(data)
    c = ReactiveMP.ar_unit(artype, order)
    return inference(
        model = lar_model(artype, n, order, c, stype, τ),
        data = (y = data,),
        initmarginals = lar_init_marginals(artype, order),
        returnvars = (γ = KeepEach(), θ = KeepEach(), x = KeepLast()),
        iterations = niter,
        free_energy = Float64
    )
end

# The following coefficients correspond to stable poles
coefs_ar_5 = [0.10699399235785655, -0.5237303489793305, 0.3068897071844715, -0.17232255282458891, 0.13323964347539288]

function generate_lar_data(rng, n, θ, γ, τ)
    order        = length(θ)
    states       = Vector{Vector{Float64}}(undef, n + 3order)
    observations = Vector{Float64}(undef, n + 3order)

    γ_std = sqrt(inv(γ))
    τ_std = sqrt(inv(γ))

    states[1] = randn(rng, order)

    for i in 2:(n + 3order)
        states[i]       = vcat(rand(rng, Normal(dot(θ, states[i - 1]), γ_std)), states[i - 1][1:(end - 1)])
        observations[i] = rand(rng, Normal(states[i][1], τ_std))
    end

    return states[(1 + 3order):end], observations[(1 + 3order):end]
end

@testset "Latent autoregressive model" begin

    # Seed for reproducibility
    rng = StableRNG(123)

    # Number of observations in synthetic dataset
    n = 500

    # AR process parameters
    real_γ = 5.0
    real_τ = 5.0
    real_θ = coefs_ar_5
    states, observations = generate_lar_data(rng, n, real_θ, real_γ, real_τ)

    # Test AR(1) + Univariate
    result = lar_inference(observations, 1, Univariate, ARsafe(), 15, real_τ)
    qs     = result.posteriors
    fe     = result.free_energy

    (γ, θ, xs) = (qs[:γ], qs[:θ], qs[:x])

    @test length(xs) === n
    @test length(γ) === 15
    @test length(θ) === 15
    @test length(fe) === 15
    @test abs(last(fe) - 518.9182342) < 0.01
    @test last(fe) < first(fe)
    @test all(filter(e -> abs(e) > 1e-3, diff(fe)) .< 0)

    # Test AR(k) + Multivariate
    for k in 1:4
        result = lar_inference(observations, k, Multivariate, ARsafe(), 15, real_τ)
        qs     = result.posteriors
        fe     = result.free_energy

        (γ, θ, xs) = (qs[:γ], qs[:θ], qs[:x])

        @test length(xs) === n
        @test length(γ) === 15
        @test length(θ) === 15
        @test length(fe) === 15
        @test last(fe) < first(fe)
    end

    # AR(5) + Multivariate
    result = lar_inference(observations, length(real_θ), Multivariate, ARsafe(), 15, real_τ)
    qs     = result.posteriors
    fe     = result.free_energy

    (γ, θ, xs) = (qs[:γ], qs[:θ], qs[:x])

    @test length(xs) === n
    @test length(γ) === 15
    @test length(θ) === 15
    @test length(fe) === 15
    @test abs(last(fe) - 514.66086) < 0.01
    @test all(filter(e -> abs(e) > 1e-1, diff(fe)) .< 0)
    @test (mean(last(γ)) - 3.0std(last(γ)) < real_γ < mean(last(γ)) + 3.0std(last(γ)))

    @test_plot "models" "lar" begin
        p1 = plot(first.(states), label = "Hidden state")
        p1 = scatter!(p1, observations, label = "Observations")
        p1 = plot!(p1, first.(mean.(xs)), ribbon = sqrt.(first.(var.(xs))), label = "Inferred states", legend = :bottomright)

        p2 = plot(mean.(γ), ribbon = std.(γ), label = "Inferred transition precision", legend = :bottomright)
        p2 = plot!([real_γ], seriestype = :hline, label = "Real transition precision")

        p3 = plot(fe, label = "Bethe Free Energy")

        p = plot(p1, p2, p3, layout = @layout([a; b c]))
    end

    @test_benchmark "models" "lar" lar_inference($observations, length($real_θ), Multivariate, ARsafe(), 15, $real_τ)
end

end
