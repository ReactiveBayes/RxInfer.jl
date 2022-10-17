module RxInferModelsULGSSMTest

using Test, InteractiveUtils
using RxInfer, BenchmarkTools, Random, Plots, Dates, LinearAlgebra, StableRNGs

# `include(test/utiltests.jl)`
include(joinpath(@__DIR__, "..", "..", "utiltests.jl"))

## Model definition
@model function multivariate_lgssm_model(n, x0, A, B, Q, P)

    # We create constvar references for better efficiency
    cA = constvar(A)
    cB = constvar(B)
    cQ = constvar(Q)
    cP = constvar(P)

    # `x` is a sequence of hidden states
    x = randomvar(n)
    # `y` is a sequence of "clamped" observations
    y = datavar(Vector{Float64}, n)

    x_prior ~ MvNormal(mean = mean(x0), cov = cov(x0))
    x_prev = x_prior

    for i in 1:n
        x[i] ~ MvNormal(mean = cA * x_prev, cov = cQ)
        y[i] ~ MvNormal(mean = cB * x[i], cov = cP)
        x_prev = x[i]
    end
end

## Inference definition
function multivariate_lgssm_inference(data, x0, A, B, Q, P)
    return inference(model = multivariate_lgssm_model(length(data), x0, A, B, Q, P), data = (y = data,), free_energy = true, options = (limit_stack_depth = 500,))
end

@testset "Linear Gaussian State Space Model" begin

    ## Data creation
    function generate_data(rng, A, B, Q, P)
        x_prev = [10.0, -10.0]

        x = Vector{Vector{Float64}}(undef, n)
        y = Vector{Vector{Float64}}(undef, n)

        for i in 1:n
            x[i] = rand(rng, MvNormal(A * x_prev, Q))
            y[i] = rand(rng, MvNormal(B * x[i], P))
            x_prev = x[i]
        end

        return x, y
    end

    # Seed for reproducibility
    rng = StableRNG(1234)
    # We will model 2-dimensional observations with rotation matrix `A`
    # To avoid clutter we also assume that matrices `A`, `B`, `P` and `Q`
    # are known and fixed for all time-steps
    θ = π / 35
    A = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    B = diageye(2)
    Q = diageye(2)
    P = 25.0 .* diageye(2)
    n = 1_000 # Number of observations
    x, y = generate_data(rng, A, B, Q, P)
    x0 = MvNormalMeanCovariance(zeros(2), 100.0 * diageye(2))
    ## -------------------------------------------- ##
    ## Inference execution
    mresult = multivariate_lgssm_inference(y, x0, A, B, Q, P)
    xmarginals = mresult.posteriors[:x]
    fe = mresult.free_energy
    ## Test inference results
    @test length(xmarginals) === n
    # We use 3.0var instead of 3.0std here for easier dot broadcasting with mean
    @test all((mean.(xmarginals) .- 3.0 .* var.(xmarginals)) .< x .< (mean.(xmarginals) .+ 3.0 .* var.(xmarginals)))
    @test all(isposdef.(cov.(xmarginals)))
    @test length(fe) === 1
    @test abs(last(fe) - 6275.9015944677) < 0.01

    ## Create output plots
    @test_plot "models" "mlgssm" begin
        slicedim(dim) = (a) -> map(e -> e[dim], a)

        subrange = 100:500
        px = plot()

        px = plot!(px, x[subrange] |> slicedim(1), label = "Hidden Signal (dim-1)", color = :orange)
        px = plot!(px, x[subrange] |> slicedim(2), label = "Hidden Signal (dim-2)", color = :green)

        px = plot!(
            px,
            mean.(xmarginals)[subrange] |> slicedim(1),
            ribbon = var.(xmarginals)[subrange] |> slicedim(1) .|> sqrt,
            fillalpha = 0.5,
            label = "Estimated Signal (dim-1)",
            color = :teal
        )
        px = plot!(
            px,
            mean.(xmarginals)[subrange] |> slicedim(2),
            ribbon = var.(xmarginals)[subrange] |> slicedim(2) .|> sqrt,
            fillalpha = 0.5,
            label = "Estimated Signal (dim-1)",
            color = :violet
        )

        return px
    end

    @test_benchmark "models" "mlgssm" multivariate_lgssm_inference($y, $x0, $A, $B, $Q, $P)
end

end
